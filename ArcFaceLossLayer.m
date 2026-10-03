classdef ArcFaceLossLayer < nnet.layer.Layer
    % This layer implements Additive Angular Margin Loss (ArcFace)
    properties (Learnable)
        % Learnable weights
        W  % Weight matrix (embeddingSize x numClasses)
    end
    
    properties
        % Hyperparameters
        Margin  % Angular margin (typically 0.5)
        Scale   % Feature scale (typically 64)
        NumClasses
    end
    
    methods
        function layer = ArcFaceLayer(numClasses, embeddingSize, margin, scale, name)
            % layer = ArcFaceLayer(numClasses, embeddingSize, margin, scale, name)
            % creates an ArcFace layer.
            
            layer.NumClasses = numClasses;
            layer.Margin = margin;
            layer.Scale = scale;
            layer.Name = name;
            layer.Description = "ArcFace layer";
            
            % Initialize weights
            layer.W = randn(embeddingSize, numClasses) * 0.01;
        end
        
        function Z = predict(layer, X)
            % X: input features (embeddingSize x batchSize)
            % W: weight matrix (embeddingSize x numClasses)
            
            % Normalize features and weights
            X_norm = X ./ vecnorm(X, 2, 1);
            W_norm = layer.W ./ vecnorm(layer.W, 2, 1);
            
            % Calculate cosine of angles
            cos_theta = W_norm' * X_norm;  % (numClasses x batchSize)
            
            % Apply scale
            Z = layer.Scale * cos_theta;  % Logits before margin
        end
        
        function [Z, memory] = forward(layer, X)
            % Forward pass for training
            X_norm = X ./ vecnorm(X, 2, 1);
            W_norm = layer.W ./ vecnorm(layer.W, 2, 1);
            cos_theta = W_norm' * X_norm;
            
            % Store for backpropagation
            memory.X_norm = X_norm;
            memory.W_norm = W_norm;
            memory.cos_theta = cos_theta;
            
            Z = layer.Scale * cos_theta;
        end
        
        function [dLdX, dLdW] = backward(layer, X, Z, dLdZ, memory)
            % Backward pass
            batchSize = size(X, 2);
            
            % Extract cached values
            X_norm = memory.X_norm;
            W_norm = memory.W_norm;
            cos_theta = memory.cos_theta;
            
            % Calculate gradients
            dLdcos_theta = layer.Scale * dLdZ;
            
            % Gradient for features
            dcosdX = (W_norm - cos_theta .* X_norm) ./ vecnorm(X, 2, 1);
            dLdX = dLdcos_theta' * dcosdX';
            
            % Gradient for weights
            dcosdW = (X_norm - cos_theta .* W_norm) ./ vecnorm(layer.W, 2, 1);
            dLdW = X_norm * dLdcos_theta';
            
            % Reshape gradients
            dLdX = reshape(dLdX, size(X));
            dLdW = reshape(dLdW, size(layer.W));
        end
    end
end