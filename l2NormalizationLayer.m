classdef l2NormalizationLayer < nnet.layer.Layer
    methods
        function layer = l2NormalizationLayer()
            layer.Name = 'l2_norm';
            layer.Description = "L2 normalization layer";
        end
        
        function Z = predict(layer, X)
            epsilon = single(1e-6);
            norm = sqrt(sum(X.^2, 1)) + epsilon;
            Z = X ./ norm;
        end
    end
end