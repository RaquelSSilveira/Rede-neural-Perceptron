clc;
clear;
pkg load nnet;

base = load('column_3C.dat');
X = base(:, 1:6); 
y = base(:, 7);

% Função para dividir os dados em conjuntos de treino e teste
function [X_treino, X_teste, y_treino, y_teste] = dividir_dados(X, y, porc_treino)
    N = size(X, 1); % Número de amostras
    indices = randperm(N); % Permutar amostras
    tam_treino = floor(porc_treino * N);
    X_treino = X(indices(1:tam_treino), :);
    y_treino = y(indices(1:tam_treino));
    X_teste = X(indices(tam_treino+1:end), :);
    y_teste = y(indices(tam_treino+1:end));
end

num_execucoes = 10;
acerto = zeros(num_execucoes, 1);
porc_treino = 0.7; % Porcentagem de dados para treino

for i = 1:num_execucoes
    % Dividir os dados
    [X_treino, X_teste, y_treino, y_teste] = dividir_dados(X, y, porc_treino);
    
    % Normalizar os dados
    [X_treino, media, desvio] = zscore(X_treino); 
    X_teste = (X_teste - media) ./ desvio;
    
    Q = 15; % Número de neurônios na camada oculta ajustado
    num_classes = length(unique(y_treino)); % Número de classes
    min_max_entrada = [zeros(size(X_treino, 2), 1), ones(size(X_treino, 2), 1)];
    
    % Criar a rede neural com 2 camadas ocultas
    rede = newff(min_max_entrada, [Q, Q, num_classes], {'logsig', 'logsig', 'logsig'}, 'trainlm');
    
    % Configurar a rede
    rede.trainParam.epochs = 1000; % Número máximo de épocas ajustado
    rede.trainParam.goal = 0.001; % Critério de parada
    rede.trainParam.lr = 0.01; % Taxa de aprendizado
    
    % Transformar os rótulos para formato de vetor binário
    y_treino_bin = full(ind2vec(y_treino')); % Converter os rótulos para formato binário
    y_teste_bin = full(ind2vec(y_teste')); % Converter os rótulos para formato binário
    
    % Treinar a rede
    [rede, ~] = train(rede, X_treino', y_treino_bin);
    
    % Fazer previsões
    y_pred = sim(rede, X_teste');
    [~, y_pred] = max(y_pred, [], 1); % Obter índices das classes previstas
    
    % Calcular a acurácia
    acerto(i) = sum(y_pred' == y_teste) / length(y_teste);
end

% Calcular a acurácia média
acerto_medio = mean(acerto);
disp(['TAXA DE ACERTO EM 10 EXECUCOES: ', num2str(acerto_medio)]);
