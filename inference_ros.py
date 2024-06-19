#!/usr/bin/env python3

import os
from PIL import Image
import rospy
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2

from models.bisenet import BiSeNet
from utils.common import vis_parsing_maps


def prepare_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    image_tensor = transform(image)
    image_batch = image_tensor.unsqueeze(0)
    return image_batch


class FaceParsingNode:
    def __init__(self):
        self.model_name = rospy.get_param("~model", "resnet34")
        self.weight_path = rospy.get_param("~weight", "./weights/resnet34.pt")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = 19
        self.model = BiSeNet(self.num_classes, backbone_name=self.model_name)
        self.model.to(self.device)

        if os.path.exists(self.weight_path):
            self.model.load_state_dict(torch.load(self.weight_path))
        else:
            raise ValueError(f"Weights not found from given path ({self.weight_path})")

        self.model.eval()

        self.bridge = CvBridge()
        self.pub = rospy.Publisher('/face_parsing/output', RosImage, queue_size=10)
        rospy.Subscriber('/d435_front/color/image_rect_color/decompressed', RosImage, self.image_callback)

    @torch.no_grad()
    def inference(self, image):
        resized_image = image.resize((512, 512), resample=Image.BILINEAR)
        transformed_image = prepare_image(resized_image)
        image_batch = transformed_image.to(self.device)

        output = self.model(image_batch)[0]
        predicted_mask = output.squeeze(0).cpu().numpy().argmax(0)

        return vis_parsing_maps(resized_image, predicted_mask, save_image=False, target_classes=['skin',
                                                                                                 'l_brow',
                                                                                                 'r_brow',
                                                                                                 'l_eye',
                                                                                                 'r_eye',
                                                                                                 'eye_g',
                                                                                                 'l_ear',
                                                                                                 'r_ear',
                                                                                                 'ear_r',
                                                                                                 'nose',
                                                                                                 'mouth',
                                                                                                 'u_lip',
                                                                                                 'l_lip',
                                                                                                 'neck',
                                                                                                 'neck_l',
                                                                                                 'hair',
                                                                                                 ])

    def image_callback(self, ros_image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
            cv_image = cv_image[100:300, 200:500]

            pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            result_image = self.inference(pil_image)

            result_cv_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            result_ros_image = self.bridge.cv2_to_imgmsg(result_cv_image, "rgb8")
            self.pub.publish(result_ros_image)
        except Exception as e:
            rospy.logerr(f"Failed to process image: {e}")


if __name__ == "__main__":
    rospy.init_node('face_parsing_node', anonymous=True)
    node = FaceParsingNode()
    rospy.spin()
