#
import torchvision
from transformers import ViTForImageClassification

class Model:

    def __init__(self, num_classes):
                
        self.num_classes = num_classes

    def build_model(self):
        
        # load pretrained model
        model_name = "google/vit-base-patch16-224"
        model = ViTForImageClassification.from_pretrained(
                    model_name,
                    num_labels=self.num_classes,
                    ignore_mismatched_sizes=True,
                )
        
        '''
        # get input feature dimension for the object classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        
        # replace the pre-trained predictor with a new head
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        # get input feature dimension for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        
        # replace the pre-trained mask predictor with a new head
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            self.num_classes
        )
        '''       

        # freeze bert layers for fine tuning
        for params in model.vit.parameters():
            params.requires_grad = False 

        return model

 
