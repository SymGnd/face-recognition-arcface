classdef ArcFaceTransformLayer < nnet.layer.Layer & nnet.layer.Formattable
    properties (Learnable)
        W  % Weight matrix (embeddingSize x numClasses)
    end
    
    properties
        Margin  % Angular margin (in radians)
        Scale   % Feature scale (e.g., 64)
        NumClasses
        EmbeddingSize
    end
    
    methods
        function layer = ArcFaceTransformLayer(numClasses, embeddingSize, margin, scale, name)
            % Constructor
            arguments
                numClasses (1,1) double
                embeddingSize (1,1) double
                margin (1,1) double = 0.5
                scale (1,1) double = 64
                name string = "arcface"
            end
    
            layer.Name = name;
            layer.Description = "ArcFace transformation layer";
            layer.Type = "ArcFace";
            layer.Margin = margin;
            layer.Scale = scale;
            layer.NumClasses = numClasses;
            layer.EmbeddingSize = embeddingSize;
    
            % Initialize weights
            layer.W = randn([embeddingSize, numClasses], 'single');
        end
    
        function Z = predict(layer, X)
            % Forward pass (prediction)
            % X: N x D (batchSize x embeddingSize)
    
            X = dlarray(X);
            X = X ./ sqrt(sum(X.^2, 2) + 1e-10);  % Normalize features
    
            W = layer.Weights;
            W = W ./ sqrt(sum(W.^2, 1) + 1e-10);  % Normalize weights
    
            % Cosine similarity
            cosine = X * W;
    
            % Multiply by scale (no margin applied here during inference)
            Z = layer.Scale * cosine;
        end
    
        function [Z, memory] = forward(layer, X)
            % Training forward pass (apply margin to true class logit)
            X_norm = X ./ vecnorm(X, 2, 1);
            W_norm = layer.W ./ vecnorm(layer.W, 2, 1);
            cos_theta = W_norm' * X_norm;
    
            % Apply margin: simulate by modifying diagonal (simplified demo)
            theta = acos(cos_theta);
            cos_theta_m = cos(theta + layer.Margin);
            cos_theta_mod = cos_theta;
    
            batchSize = size(X, 2);
            for i = 1:min(batchSize, layer.NumClasses)
                cos_theta_mod(i, i) = cos_theta_m(i, i);
            end
    
            Z = layer.Scale * cos_theta_mod;
    
            memory.X_norm = X_norm;
            memory.W_norm = W_norm;
            memory.cos_theta = cos_theta;
        end
    
        function [dLdX, dLdW] = backward(layer, X, ~, dLdZ, memory)
            % Backprop: simplified approximation (angular margin not exact here)
            X_norm = memory.X_norm;
            W_norm = memory.W_norm;
    
            dLdX = layer.W * dLdZ;
            dLdW = X * dLdZ';
    
            % Normalize gradients
            dLdX = dLdX - X_norm .* sum(X_norm .* dLdX, 1);
            dLdW = dLdW - W_norm .* sum(W_norm .* dLdW, 1);
    
            dLdX = layer.Scale * dLdX;
            dLdW = layer.Scale * dLdW;
        end
    end

end