clc;
clear;

% Solicitar valores de entrada e rótulos
v = input('Digite o número de vetores: ');

% Matriz de entradas
X = zeros(v, 2);
y = zeros(v, 1);

for i = 1:v
    X(i, :) = input(sprintf('Digite o vetor x%d como [x1, x2]: ', i));
    y(i) = input(sprintf('Digite o rótulo para o vetor x%d (0 ou 1): ', i));
end

% Normalização dos vetores de entrada
X = zscore(X);

% Adicionar coluna de 1s (bias)
X = [ones(v, 1), X];

% Inicializar pesos aleatoriamente
sz = size(X, 2); % Número de características incluindo o bias
w = rand(sz, 1) * 0.1; % Inicializar pesos para uma saída
n = 0.1; % Taxa de aprendizado
epocas_max = 100;
erro_max = 0.05; % Tolerância para taxa de erro

% Função de ativação
function output = ativa(u)
    output = u > 0; % Retorna 1 se u > 0, retorna 0 se u <= 0
end

% Função de plotagem
function plotar(X, y, w, epoca)
    figure(1);
    clf;
    hold on;

    pos = y == 1;
    neg = y == 0;

    plot(X(pos, 2), X(pos, 3), 'r*', 'MarkerSize', 8, 'LineWidth', 2);
    plot(X(neg, 2), X(neg, 3), 'bx', 'MarkerSize', 8, 'LineWidth', 2);

    % Desenhar a reta de decisão
    x_vals = linspace(min(X(:, 2)), max(X(:, 2)), 100);
    y_vals = (-w(1) - w(2) * x_vals) / w(3);
    plot(x_vals, y_vals, 'k-', 'LineWidth', 2);

    xlabel('x1');
    ylabel('x2');
    title(sprintf('Rede Neural Perceptron - Época %d', epoca));
    legend('Classe 1', 'Classe 0', 'Reta de Decisão');
    hold off;
    drawnow;
end

% Função de treinamento
function [w, convergiu] = treino(X, y, w, n, epocas_max, erro_max)
    m = size(X, 1);
    convergiu = false;
    for iter = 1:epocas_max
        % Embaralhar os índices para não enviesar o treino
        indices = randperm(m);
        X_shuffled = X(indices, :);
        y_shuffled = y(indices);

        % Contar erros
        num_errors = 0;

        for i = 1:m
            % Calcular a ativação (u = w1*x1 + w2*x2 + w0)
            u = X_shuffled(i, :) * w;

            % Calcular a saída (y(t))
            saida = ativa(u);

            % Calcular o erro (e = d - y)
            e = y_shuffled(i) - saida; % Erro esperado
            w = w + n * e * X_shuffled(i, :)'; % Atualização dos pesos

            % Contar erros
            num_errors = num_errors + (e ~= 0);
        end

        % Plotar a cada época
        plotar(X, y, w, iter);

        % Calcular e verificar a taxa de erro
        error_rate = num_errors / m;
        if error_rate < erro_max
            convergiu = true;
            fprintf('Treinamento convergiu na época %d com taxa de erro %.2f%%\n', iter, error_rate * 100);
            break;
        end
    end
end

% Função para calcular a porcentagem de acerto
function [acerto] = calcularPrecisao(X, y, w)
    previsao = arrayfun(@(i) ativa(X(i, :) * w), 1:size(X, 1))';
    num_correct = sum(previsao == y);
    acerto = (num_correct / length(y)) * 100;
end

% Treinamento
[w, convergiu] = treino(X, y, w, n, epocas_max, erro_max);

if ~convergiu
    fprintf('Treinamento não convergiu após o número máximo de épocas.\n');
end

% Calcular e exibir a porcentagem de acerto
acerto = calcularPrecisao(X, y, w);
fprintf('Porcentagem de acerto final: %.2f%%\n', acerto);

