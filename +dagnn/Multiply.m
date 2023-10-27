classdef Multiply < dagnn.ElementWise
  %SUM DagNN sum layer
  %   The SUM layer takes the sum of all its inputs and store the result
  %   as its only output.

  properties (Transient)
    numInputs
  end

  methods
    function outputs = forward(obj, inputs, params)
%       obj.numInputs = numel(inputs) ;
      obj.numInputs = 2 ;
%       outputs{1} = inputs{1} ;
      
       A=inputs{1,1};
       B=inputs{1,2};
       [ mm1 mm2 mm3 mm4]=size(A);
       for i=1:mm3
           for ii=1:mm4
               c(:,:,i,ii)=A(:,:,i,ii)*B(:,:,i,ii);
           end
       end
       outputs{1}=c;
      end
    

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      for k = 1:obj.numInputs
        derInputs{k} = derOutputs{1} ;
      end
      derParams = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1} = inputSizes{1} ;
      for k = 2:numel(inputSizes)
        if all(~isnan(inputSizes{k})) && all(~isnan(outputSizes{1}))
          if ~isequal(inputSizes{k}, outputSizes{1})
            warning('Multiply layer: the dimensions of the input variables is not the same.') ;
          end
        end
      end
    end

    function rfs = getReceptiveFields(obj)
      numInputs = numel(obj.net.layers(obj.layerIndex).inputs) ;
      rfs.size = [1 1] ;
      rfs.stride = [1 1] ;
      rfs.offset = [1 1] ;
      rfs = repmat(rfs, numInputs, 1) ;
    end

    function obj = Multiply(varargin)
      obj.load(varargin) ;
    end
  end
end
