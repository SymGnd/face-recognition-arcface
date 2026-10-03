classdef CenterLossLayer < nnet.layer.RegressionLayer
    properties
        Lambda
        NumClasses
        Centers
    end
    
    methods
        function layer = CenterLossLayer(numClasses, lambda, name)
            layer.NumClasses = numClasses;
            layer.Lambda = lambda;
            layer.Name = name;
            layer.Description = "Center loss";
            
            % Initialize centers (not as learnable parameters)
            layer.Centers = zeros(128, numClasses); % Adjust 128 to your embedding size
        end
        
        function loss = forwardLoss(layer, Y, T)
            % Y: embeddings (embeddingSize x batchSize)
            % T: class labels (1 x batchSize)
            
            % Convert one-hot to indices if needed
            if size(T,1) > 1
                [~, classIdx] = max(T, [], 1);
            else
                classIdx = T;
            end
            
            % Get centers for current batch
            batchCenters = layer.Centers(:,classIdx);
            
            % Calculate center loss
            diff = Y - batchCenters;
            centerLoss = 0.5 * sum(diff(:).^2) / size(Y,2);
            loss = layer.Lambda * centerLoss;
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % Convert one-hot to indices if needed
            if size(T,1) > 1
                [~, classIdx] = max(T, [], 1);
            else
                classIdx = T;
            end
            
            batchSize = size(Y,2);
            
            % Calculate gradient for embeddings
            dLdY = layer.Lambda * (Y - layer.Centers(:,classIdx)) / batchSize;
            
            % Update centers manually (since we can't use Learnable)
            alpha = 0.01; % Center learning rate
            for i = 1:batchSize
                c = classIdx(i);
                layer.Centers(:,c) = layer.Centers(:,c) - alpha * (Y(:,i) - layer.Centers(:,c));
            end
        end
    end
end