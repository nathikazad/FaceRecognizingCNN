function TrainFaces(varargin)
facesDir = dir('chosen_faces');
i = 1;
l = 1;
for file = facesDir'
    if(file.name(1)=='.') continue; end
    indFaces = dir(fullfile('chosen_faces',file.name,'*.ppm'));
    for face = indFaces'
        faces.images.data(:,:,:,i) = single(imread(fullfile('chosen_faces',file.name,face.name)));
        faces.images.labels(i) = l;
        i = i + 1;
    end
    set_index = (l-1)*50;
    faces.images.set(set_index+1:set_index+45) = 1;
    faces.images.set(set_index+46:set_index+50) = 2;
    l=l+1;
end


trainOpts.batchSize = 60 ;
trainOpts.numEpochs = 12 ;
trainOpts.continue = true ;
trainOpts.gpus = [] ;
trainOpts.learningRate = 0.001 ;
trainOpts.expDir = 'data/face-experiment' ;

faces.images.mean = mean(faces.images.data(:)) ;
faces.images.data = faces.images.data - faces.images.mean ;
net = initializeFacesCNN();

[net,info] = cnn_train(net, faces, @getBatch, trainOpts) ;

figure(2) ; clf ; colormap gray ;
vl_imarraysc(squeeze(net.layers{1}.weights{1}),'spacing',2);
[x,y,c,f]=size(net.layers{1, 1}.weights{1});
axis equal ; title(sprintf('%dx%dx%dx%d filters in the first layer',x,y,c,f) ...
    ,'FontSize',20) ;

% --------------------------------------------------------------------
function [im, labels] = getBatch(faces, batch)
% --------------------------------------------------------------------
im = faces.images.data(:,:,:,batch) ;
%im = 256 * reshape(im, 64, 64, 1, []) ;
labels = faces.images.labels(1,batch) ;
