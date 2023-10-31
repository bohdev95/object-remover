import os

import cv2
import numpy as np
from PIL import Image

from . import MaskFormer, LaMa
from .utils import package_path, save_image
from typing import List
import json

class SemanticObjectRemover:
    def __init__(self, lama_ckpt: str, lama_config: str,
                 maskformer_ckpt: str, label_file: str):
        self.maskformer = MaskFormer(maskformer_ckpt, label_file)
        self.lama = LaMa(lama_ckpt, lama_config)

    @staticmethod
    def __load_image_to_array(image_path: str) -> np.ndarray:
        img = Image.open(image_path)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        return np.array(img)

    @staticmethod
    def __array_to_image(img_array: np.ndarray) -> Image:
        return Image.fromarray(img_array.astype(np.uint8))

    @staticmethod
    def __dilate_mask(mask: np.ndarray, dilate_factor: int = 15) -> np.ndarray:
        mask = mask.astype(np.uint8)
        mask = cv2.dilate(mask, np.ones((dilate_factor, dilate_factor), np.uint8), iterations=1)
        return mask
    @staticmethod
    def filter_points(new_points):
        result = []
        
        for i, point in enumerate(new_points):
            
            if i > 0 and i < len(new_points) - 1:
                previous_index = i - 1
                next_index = i + 1

                first_point = new_points[previous_index]
                second_point = point
                third_point = new_points[next_index]

                if second_point[1] - third_point[1] > 20:
                    result.append(point)
                    result.append((third_point[0] - 1, second_point[1]))
                elif third_point[1] - second_point[1] > 20:
                    result.append(point)
                    result.append((third_point[0], second_point[1]))
                elif abs(second_point[1] - first_point[1]) > 20:
                    result.append(point)
                elif (third_point[0] - first_point[0]) < 50 and abs(second_point[1] - first_point[1]) < 10 and third_point[1] == first_point[1]:
                    continue
                elif abs((second_point[0] - first_point[0]) - (third_point[0] - second_point[0])) < 5 and abs((second_point[1] - first_point[1]) - (third_point[1] - second_point[1])) < 5:
                    continue
                elif (third_point[0] - second_point[0]) < 5 and abs(third_point[1] - second_point[1]) < 10:
                    continue
                else:
                    result.append(point)
                
            else:
                result.append(new_points[i])

        result.sort()
        
        return result
    @staticmethod
    def __generate_mask(mask_image, distance) -> np.ndarray:
        """ Remove objects specified by labels from input image. """
        my_array = []
        
        for i, x in enumerate(mask_image[0]):
            for j, y in enumerate(mask_image):
                if mask_image[j][i] > 0:
                    if len(my_array) > 0:
                        if my_array[len(my_array) - 1][1] == j:
                            break 
                    my_array.append((distance + i, j))
                    break

        return my_array
    @staticmethod
    def __generate_window(mask_image, distance) -> np.ndarray:
        my_array = []
        y = 0
        for horizen in mask_image:
            x = 0
            for pix in horizen:
                if pix == 255 :
                    my_array.append((distance + x, y))
            
                x = x + 1
            y = y + 1

        my_array.sort()
        
        points_2d = []
        
        unique_x_values = set(point[0] for point in my_array)
        
        y = 0
        for x in unique_x_values:
            
            points = []
            for point in my_array:
                if x == point[0]:
                    points.append(point)
            points.sort()


            if len(points) > 100:
                points_2d.append(points[0])
                count = 0

                for i, p in enumerate(points):
                    if i == len(points) - 1:
                        points_2d.append(p)
                    if p[1] == points_2d[len(points_2d) - 1][1] + count:
                        count += 1
                        continue
                    elif p[1] - ( points_2d[len(points_2d) - 1][1] + count) < 100:
                        count += p[1] - ( points_2d[len(points_2d) - 1][1] + count)
                        continue
                    else:
                        points_2d.append(points[i - 1])
                        points_2d.append(p)
                        count = 0

        points_2d.sort()
        new_points = []
        
        for i, point in enumerate(points_2d[1::2]):

            previous_index = i * 2

            first_point = points_2d[previous_index]
            second_point = point
 
            if abs(first_point[1] - second_point[1]) < 100:
                continue
            elif len(new_points) == 0:          
                data = {
                    "points":[],
                    "third_point": [], 
                    "fourth_point": []
                }
                data["points"].append(first_point)
                data["points"].append(second_point)

                new_points.append(data)

            else:
                is_avail = True
                for i, p in enumerate(new_points):
                    if len(p["third_point"]) == 0:
                   
                        if first_point[0] - p["points"][0][0] == 1 and abs(first_point[1] - p["points"][0][1]) < 10:
                            new_points[i]["third_point"] = first_point
                            new_points[i]["fourth_point"] = second_point
                            is_avail = False
                            break
                        
                    else:
                        if first_point[0] - p["third_point"][0] == 1 and abs(first_point[1] - p["third_point"][1]) < 10:
                            new_points[i]["third_point"] = first_point
                            new_points[i]["fourth_point"] = second_point
                            is_avail = False
                            break
                
                if is_avail:
                    data = {
                        "points":[],
                        "third_point": [], 
                        "fourth_point": []
                    }
                    data["points"].append(first_point)
                    data["points"].append(second_point)
                    new_points.append(data)                 

            

        return new_points
    def remove_objects_from_image(self, input_path: str , labels: List[str], dilate_kernel_size: int = 15) -> Image:
        
        if not os.path.exists(input_path) or os.path.isdir(input_path):
            raise IOError(f"{input_path} is not a file.")

        # Load the image from file into ndarray
        image = self.__load_image_to_array(input_path)

        mask = self.__load_image_to_array('mask/mask.jpg')

        mask = self.__dilate_mask(mask, dilate_kernel_size)

        image = self.lama.inpaint(image, mask)
  
        return self.__array_to_image(image)

    def create_mask_from_image(self, input_path: str , labels: List[str], dilate_kernel_size: int = 15) -> Image:
        
        if not os.path.exists(input_path) or os.path.isdir(input_path):
            raise IOError(f"{input_path} is not a file.")

        # Load the image from file into ndarray
        image = self.__load_image_to_array(input_path)

        masks, labels = self.maskformer.segment(image, labels)
        

        with open("mask/resolution.json", 'r') as f:
            resolutions = json.load(f)
        t_width = int(resolutions["width"])
        t_height = int(resolutions["height"])

        img = Image.open(input_path)
        w, h = img.size

        if t_height * w / h < t_width:
            t_width = int(t_height * w / h)

        distance = int((resolutions["width"] - t_width)/2)

        # # Save original image if no objects were found to remove
        if masks is None and labels is None:
            blackblankimage = np.zeros(shape=[512, 512, 3], dtype=np.uint8)
            data = {
                "width" : int(resolutions["width"]),
                "height" : int(resolutions["height"]),
                "floor": [[distance + 1, int(t_height/2)], [t_width - 1, int(t_height/2)]],
                "window": []
            }
            with open('mask/data.json', 'w') as f:
                # Convert the array to JSON and write it to the file
                json.dump(data, f)
            return self.__array_to_image(blackblankimage)
               

        # # Dilate mask to avoid unmasked edge effect
        if dilate_kernel_size > 0:
            masks = [self.__dilate_mask(mask, dilate_kernel_size) for mask in masks]

        # # Loop over masks and do in-painting for each selected label
        
        
        floor = []
        window = []
        land = []

        # if labels[0] == "windowpane":
        #     window = masks[0]
        #     if len(masks) > 1:
        #         floor = masks[1]
        #     else:
        #         floor = []
        for mask, label in zip(masks, labels):
            if label == "windowpane" or label == "glass":
                window = mask
            elif label == "floor" or label == "rug" or label == "wall":
                floor = mask
            else:
                land = mask
        
        for mask, label in zip(masks, labels):
            if label == "floor" or label == "rug" or label == "wall":
                floor = cv2.add(floor, mask)
            elif label == "windowpane" or label == "glass":
                window = cv2.add(window, mask)
            else:
                land = cv2.add(land, mask)


        if len(land) > 0:
            floor = land
                 
        if len(floor) > 0:
            floor_image = self.__array_to_image(floor)
            floor_image_resized = floor_image.resize((t_width, t_height))
            floor = np.array(floor_image_resized)
            new_points = self.__generate_mask(floor, distance)
            floor_points = self.filter_points(new_points)
        else:
            floor_points = [[distance + 1, int(t_height/2)], [t_width - 1, int(t_height/2)]]

        if len(window) > 0:
            window_image = self.__array_to_image(window)
            window_image_resized = window_image.resize((t_width, t_height))

            window = np.array(window_image_resized)
            window_points = self.__generate_window(window, distance)
        else:
            window_points = []       
        
        
        data = {
            "width" : int(resolutions["width"]),
            "height" : int(resolutions["height"]),
            "floor": floor_points,
            "window": window_points
        }

        with open('mask/data.json', 'w') as f:
            # Convert the array to JSON and write it to the file
            json.dump(data, f)

        return self.__array_to_image(floor)