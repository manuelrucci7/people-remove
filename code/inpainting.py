from migan.migan_inference import Generator as MIGAN
from migan.migan_utils import preprocess, resize, read_mask
import torch
import numpy as np
import cv2
from PIL import Image

class MiganInpainting:
    def __init__(self, model_path, size=[512, 512]):
        self.model = MIGAN(resolution=size[0])
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.size = size[0]
        self.device = torch.device("cuda")
        self.model = self.model.to(self.device)

    def inpaint(self, img, mask):
        #dilate mask
        kernel = np.ones((5,5),np.uint8)
        maskf = cv2.dilate(mask, kernel, iterations=2)
        maskf[maskf > 0] = 255
        mask_bgr = cv2.cvtColor(maskf, cv2.COLOR_GRAY2BGR)
        mask_bgr_inv = cv2.bitwise_not(mask_bgr)

        image_holes = cv2.bitwise_and(img, mask_bgr_inv)
 
        img = Image.fromarray(img)#(file_path).convert("RGB")
        img_resized = resize(img, max_size=self.size)
        mask = read_mask(maskf, invert=1)
        mask_resized = resize(mask, max_size=self.size, interpolation=Image.NEAREST)

        x = preprocess(img_resized, mask_resized, self.size)
        x = x.to(self.device)

        with torch.no_grad():
            result_image = self.model(x)[0]

        result_image = (result_image * 0.5 + 0.5).clamp(0, 1) * 255
        result_image = result_image.to(torch.uint8).permute(1, 2, 0).detach().to("cpu").numpy()

        result_image = cv2.resize(result_image, dsize=img_resized.size, interpolation=cv2.INTER_CUBIC)
        mask_resized = np.array(mask_resized)[:, :, np.newaxis] // 255
        img_resized = cv2.resize(np.array(img_resized), (result_image.shape[1], result_image.shape[0]))
        mask_resized = cv2.resize(mask_resized, (result_image.shape[1], result_image.shape[0]))

        # Ensure mask_resized has the same number of channels as img_resized and result_image
        mask_resized = np.stack([mask_resized]*3, axis=-1)

        composed_img = img_resized * mask_resized + result_image * (1 - mask_resized)
        composed_img = Image.fromarray(composed_img)

        composed_img = np.array(composed_img)
        #composed_img = cv2.cvtColor(composed_img, cv2.COLOR_RGB2BGR)
        height, width, _ = image_holes.shape

        # Resize composed_img to match the shape of image_holes
        composed_img_resized = cv2.resize(composed_img, (width, height), interpolation=cv2.INTER_CUBIC)

        res = cv2.bitwise_and(mask_bgr, composed_img_resized)
        image_filled = cv2.bitwise_or(res, image_holes)

        return image_filled
    
if __name__ == "__main__":
    # Read Image
    filepath = "images/bus.jpg"
    im = cv2.imread(filepath, cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #mask = cv2.imread("images/mask_det.jpg", cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread("images/mask_seg.jpg", cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (im.shape[1], im.shape[0]))

    # Inpainting
    #device = torch.device("cuda")
    model_path = "models/migan_512_places2.pt"
    inpainting = MiganInpainting(model_path)
    result = inpainting.inpaint(im, mask)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite("images/result_seg.jpg", result)    
    #cv2.imshow("result", result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()