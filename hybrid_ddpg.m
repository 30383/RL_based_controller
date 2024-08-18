clear;
clc;
%%
% GLOBAL PARAMETERS 
% Parameter values
num_episodes = 1024;
numValidationExperiments = 20;

%%
% Buck Boost Converter Parameters
V_source_value = 48;
L_inductance = 10e-6; 
C_capacitance = 40e-3;
R_load = 100;
%%
gain_k = 100;
integral_I = 350000;
period_val = 0.00001;
pw_percent = 50;

%%
% Signal Processing Parameters
prev_time = 0;
init_action = 1; 
stopping_criterion = 1000;
threshold1= 0.4;
threshold2 =1;
error_threshold = 0.02;
%%
Ts = 0.00001;
Tf = 0.3;
V_ref = 110; %30;%80%110;

%%
% RL Parameters
miniBatch_percent = 0.8;
learnRateActor = 0.001;
learnRateCritic = 0.001;
discountFactor = 0.99;
targetSmoothFactor = 0.99;
targetUpdateFrequency = 1;
noiseOptions.Mean = 0;
noiseOptions.Variance = 0.1;
noiseOptions.DecayRate = 1e-5;

max_steps = ceil(Tf/Ts);
ExperienceHorisonLength = 10;
ClipFactorVal = 0.2;
EntropyLossWeightVal = 0.05;
MiniBatchSizeVal = ceil(ExperienceHorisonLength*miniBatch_percent); 
NumEpochsVal = 5; 
DiscountFactorVal = 0.99;

%%

% RL Agent

mdl = 'DCDC_BBC_hybrid1';
open_system(mdl)
agentblk = [mdl '/RL Agent'];

numObs = 3; % [v0, e, de/dt]
observationInfo = rlNumericSpec([numObs,1],...
    'LowerLimit',[-inf -inf 0]',...
    'UpperLimit',[0.1 V_ref inf]');
observationInfo.Name = 'observations';
observationInfo.Description = 'integrated error, error, and measured height';
numObservations = observationInfo.Dimension(1);

a = [0;1]; 
actionInfo = rlNumericSpec([1, numel(a)], 'LowerLimit', 0, 'UpperLimit', 1);


env = rlSimulinkEnv(mdl,agentblk,observationInfo,actionInfo);

env.ResetFcn = @(in) setVariable(in,'init_action',1);
num_inputs = numObs;        
 
% Define separate input layers for observations and actions in the critic network
observationInputLayer = imageInputLayer([numObs 1 1], 'Normalization', 'none', 'Name', 'observation');
actionInputLayer = imageInputLayer([numel(a) 1 1], 'Normalization', 'none', 'Name', 'action');

% Define the hidden layers for the critic network
hiddenLayerSizes = [256, 256];
hiddenLayers = [
    fullyConnectedLayer(hiddenLayerSizes(1), 'Name', 'CriticFC1')
    reluLayer('Name', 'CriticRelu1')
    fullyConnectedLayer(hiddenLayerSizes(2), 'Name', 'CriticFC2')
    reluLayer('Name', 'CriticRelu2')
];

% Define the output layer for the critic network
outputLayer = fullyConnectedLayer(1, 'Name', 'CriticOutput');

% Connect the input layers to the hidden layers
observationBranch = observationInputLayer;
actionBranch = actionInputLayer;

% Concatenate the branches
concatenatedInputLayer = additionLayer(2, 'Name', 'concatenatedInput', 'Layers', {observationBranch, actionBranch});
% Combine the layers into a layer graph
criticLayers = [
    observationBranch
    actionBranch
    concatenatedInputLayer
    hiddenLayers
    outputLayer
];

% Create the layer graph
lgraph = layerGraph(criticLayers);

% Create the Q-value function
critic = rlQValueFunction(lgraph, {observationInfo, actionInfo});




actorNetwork = [
    imageInputLayer([numObs 1 1],'Normalization','none','Name','observation')
    fullyConnectedLayer(256, 'Name', 'ActorFC1')
    reluLayer('Name','ActorRelu1')
    fullyConnectedLayer(256, 'Name', 'ActorFC2')
    reluLayer('Name','ActorRelu2')
    fullyConnectedLayer(numAct, 'Name', 'ActorOutput')
    tanhLayer('Name', 'actorTanh')];

actorOpts = rlRepresentationOptions('LearnRate',learnRateActor,'GradientThreshold',1);

actor = rlDeterministicActorRepresentation(actorNetwork,observationInfo,actionInfo,...
    'Observation',{'observation'},actorOpts);

agentOpts = rlDDPGAgentOptions(...
    'SampleTime', Ts, ...
    'TargetSmoothFactor', targetSmoothFactor, ...
    'ExperienceBufferLength', ExperienceHorisonLength, ...
    'DiscountFactor', discountFactor, ...
    'MiniBatchSize', MiniBatchSizeVal, ...
    'NumStepsToLookAhead', 1, ...
    'ResetExperienceBufferBeforeTraining', true, ...
    'SaveExperienceBufferWithAgent', true, ...
    'ExperienceBufferSampling', 'recent', ...
    'TargetUpdateFrequency', targetUpdateFrequency, ...
    'Critics', critic, ...
    'Actors', actor, ...
    'AgentExecutionOptions', agentOpts);

agent = rlDDPGAgent(actor, critic, agentOpts);

trainOpts = rlTrainingOptions(...
    'MaxEpisodes', num_episodes, ...
    'MaxStepsPerEpisode', max_steps, ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue', inf, ...
    'ScoreAveragingWindowLength', 50, ...
    'SaveAgentCriteria', "EpisodeReward", ...
    'SaveAgentValue', 50000, ...
    'Verbose', true);


%%

% Train Agent

trainingStats = train(agent, env, trainOpts);
