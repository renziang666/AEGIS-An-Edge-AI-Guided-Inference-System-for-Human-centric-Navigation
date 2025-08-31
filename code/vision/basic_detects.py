# basic_detects.py

import numpy as np
import json
import sophon.sail as sail
import logging
import tempfile
import cv2
import os
from abc import abstractmethod
from typing import Union, List

from postprocess_detect import PostProcess as PostProcessDetect
from postprocess_segment import PostProcess as PostProcessSeg

class YoloDetect:
    # 只定义了YOLO常见的预处理和后处理部分（标准实现），继承后请补充常见方法。
    def __init__(
        self,
        bmodel_path: str,
        dev_id: int = 0,
        use_resize_padding: bool = False, # 是否启用填充版
        use_vpp: bool = True # 是否使用vpp加速
    ):
        self.bmodel_path = bmodel_path #模型路径
        self.dev_id = dev_id #算能专用设备的索引
        self.handle = sail.Handle(self.dev_id) #设备句柄
        self.bmcv = sail.Bmcv(self.handle) #bmcv处理类

        # 模型加载
        self.net = sail.Engine(
            self.bmodel_path,
            self.dev_id,
            sail.IOMode.SYSO #内存共享标志
        )
        self.graph_name = self.net.get_graph_names()[0]

        # 配置输入
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.input_dtype = self.net.get_input_dtype(self.graph_name, self.input_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.input_shapes = {self.input_name: self.input_shape}
        self.input_tensor = sail.Tensor(
            self.handle,
            self.input_shape,
            self.input_dtype,
            False, #不在系统分配内存，因为不需要从CPU侧主动输入和读取数据
            False #不在设备分配内存，因为bm_image_to_tensor函数会自动在设备分配内存
        )
        self.input_tensors = {self.input_name: self.input_tensor}

        # 配置输出
        self.output_name = self.net.get_output_names(self.graph_name)[0]
        self.output_dtype = self.net.get_output_dtype(self.graph_name, self.output_name)
        self.output_shape = self.net.get_output_shape(self.graph_name, self.output_name)
        self.output_tensor = sail.Tensor(
            self.handle,
            self.output_shape,
            self.output_dtype,
            True, #在系统分配内存，因为需要.asnumpy()方法直接读取数据
            True #在设备分配内存，因为需要传入.net.process()向前
        )
        self.output_tensors = {self.output_name: self.output_tensor}
        
        # 其余配置
        self.input_scale = self.net.get_input_scale(self.graph_name, self.input_name)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        self.ab = [x * self.input_scale / 255. for x in [1, 0, 1, 0, 1, 0]]
        self.use_resize_padding = use_resize_padding
        self.use_vpp = use_vpp

        self.cls_num = self.output_shape[2] - 4
        
        logging.info("模型已成功装载进入TPU设备内存。")

    def __del__(self):
        logging.info("模型已成功从TPU设备内存中卸载。")

    def convert_to_bmimage(self, np_img: List[str]) -> sail.BMImage:
        """
        将 numpy 图像帧转换为 BMImage 对象，内部先写文件再用 sail.Decoder 转 BMImage。
        后期加速可以搭建rtsp服务器，用推流方法。
        """

        tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_file_path = tmp_file.name
        tmp_file.close()

        success = cv2.imwrite(tmp_file_path, np_img)
        if not success:
            raise RuntimeError("写临时文件失败，无法将 numpy 图像写成 JPEG")

        try:
            decoder = sail.Decoder(tmp_file_path, True, self.dev_id)
            bmimg = sail.BMImage()
            handle = sail.Handle(self.dev_id)
            if decoder.read(handle, bmimg) != 0:
                raise RuntimeError("Decoder 解码失败，可能是写入的临时文件有问题")

            return bmimg

        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    def preprocess_bmcv(self, input_bmimg: sail.BMImage) -> sail.BMImage:
        '''
            描述了如何使用BMCV接口预处理图片。
        '''
        rgb_planar_img = sail.BMImage(
            self.handle,
            input_bmimg.height(),
            input_bmimg.width(),
            sail.Format.FORMAT_RGB_PLANAR,
            sail.DATA_TYPE_EXT_1N_BYTE
        )
        self.bmcv.convert_format(input_bmimg, rgb_planar_img)

        resized_img_rgb, transform_info = self.resize_bmcv(rgb_planar_img)
        preprocessed_bmimg = sail.BMImage(
            self.handle,
            self.net_h,
            self.net_w,
            sail.Format.FORMAT_RGB_PLANAR,
            self.img_dtype
        )
        self.bmcv.convert_to(
            resized_img_rgb,
            preprocessed_bmimg,
            ((self.ab[0], self.ab[1]),
            (self.ab[2], self.ab[3]),
            (self.ab[4], self.ab[5]))
        )
        return preprocessed_bmimg, transform_info
    
    def resize_bmcv(self, bmimg: sail.BMImage) -> tuple[sail.BMImage, dict]:
        """
            描述了如何使用BMCV接口调整图片。
        """
        img_w = bmimg.width()
        img_h = bmimg.height()
        if self.use_resize_padding:
            r_w = self.net_w / img_w
            r_h = self.net_h / img_h
            r = min(r_w, r_h)
            tw = int(round(r * img_w))
            th = int(round(r * img_h))
            tx1, ty1 = self.net_w - tw, self.net_h - th  # wh padding

            tx1 /= 2  # divide padding into 2 sides
            ty1 /= 2

            attr = sail.PaddingAtrr()
            attr.set_stx(int(round(tx1 - 0.1)))
            attr.set_sty(int(round(ty1 - 0.1)))
            attr.set_w(tw)
            attr.set_h(th)
            attr.set_r(114)
            attr.set_g(114)
            attr.set_b(114)
            
            preprocess_fn = self.bmcv.vpp_crop_and_resize_padding if self.use_vpp else self.bmcv.crop_and_resize_padding
            resized_img_rgb = preprocess_fn(
                bmimg,
                0,
                0,
                img_w,
                img_h,
                self.net_w,
                self.net_h,
                attr,
                sail.bmcv_resize_algorithm.BMCV_INTER_LINEAR
            )

            transform_info = {
                "ratio": (r, r),
                "txy": (tx1, ty1),
                "org_size": (img_w, img_h),
                "resize_type": "letterbox"
            }

        else:
            r_w = self.net_w / img_w
            r_h = self.net_h / img_h

            preprocess_fn = self.bmcv.vpp_resize if self.use_vpp else self.bmcv.resize
            resized_img_rgb = preprocess_fn(bmimg, self.net_w, self.net_h)

            transform_info = {
                "ratio": (r_w, r_h),
                "txy": (0.0, 0.0),
                "org_size": (img_w, img_h),
                "resize_type": "stretch"
            }

        return resized_img_rgb, transform_info

    def preprocess(
            self,
            imgs: List[Union[str, sail.BMImage, np.ndarray]]
    ) -> List[dict]:

        # 如果是由文件路径构成的列表
        if isinstance(imgs, list) and isinstance(imgs[0], str):
            bmimgs = []
            for img_path in imgs:
                decoder = sail.Decoder(img_path, True, self.dev_id)
                bmimg = sail.BMImage()
                handle = sail.Handle(self.dev_id)
                if decoder.read(handle, bmimg) != 0:
                    logging.info(f"解码器在尝试解码 {img_path} 时失败，请检查路径。")
                    continue
                bmimgs.append(bmimg)
        # 如果已经是BMImage
        elif isinstance(imgs, list) and isinstance(imgs[0], sail.BMImage):
            bmimgs = imgs
        # 如果是numpy数组
        elif isinstance(imgs, list) and hasattr(imgs[0], "shape"):
            bmimgs = [self.convert_to_bmimage(np_img=np_img) for np_img in imgs]
        # 如果什么都不是
        else:
            raise TypeError("imgs 类型不支持，必须是路径列表、BMImage列表或numpy图像帧列表")
        
        # 调用bmcv处理图片
        processed_bmimgs, transform_infos = [], []
        for bmimg in bmimgs:
            bmimg, transform_info = self.preprocess_bmcv(bmimg)
            processed_bmimgs.append(bmimg)
            transform_infos.append(transform_info)
        bmimgs = processed_bmimgs

        # 对batch_size>1的批次采用批处理推理类
        if self.batch_size == 1:
            bmimg = bmimgs[0]
            self.bmcv.bm_image_to_tensor(bmimg, self.input_tensor)
        else:
            BMImageArray = getattr(sail, f'BMImageArray{self.batch_size}D', None)
            if BMImageArray is None:
                raise ValueError(f"未找到对应的BMImageArray{self.batch_size}D类，请检查batch_size是否合法。")
            bmimg_array = BMImageArray()

            if len(bmimgs) > len(bmimg_array):
                raise ValueError(f"图片过多。最多接受{len(bmimg_array)}张图片，实际收到了{len(bmimgs)}张图片。")
            cvt_num = np.min((len(bmimgs), len(bmimg_array)))

            for i in range(len(cvt_num)):
                bmimg_array[i] = bmimgs[i].data()
            self.bmcv.bm_image_to_tensor(bmimg_array, self.input_tensor)
        
        return transform_infos

    def convert_centre_to_corner(self, batches: np.ndarray) -> np.ndarray:
        cx = batches[:,:,0]
        cy = batches[:,:,1]
        w = batches[:,:,2]
        h = batches[:,:,3]

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        transformed_batches = np.stack([x1, y1, x2, y2], axis=-1)
        remaining_batch = batches[:, :, 4:]

        return np.concatenate((transformed_batches, remaining_batch), axis=-1)

    def specify(
        self,
        batch: np.ndarray,
        conf_thres: float
    ) -> np.ndarray:
            inds = np.max(batch[:, 4:], axis=-1) > conf_thres
            subbatch = batch[inds]
            boxes = subbatch[:, 0:4]
            clss = np.argmax(subbatch[:, 4:], axis=-1)
            confs = np.max(subbatch[:, 4:], axis=-1)
            return np.concatenate((boxes, clss[:, None], confs[:, None]), axis=-1)

    def classify(self, boxes: np.ndarray) -> List[np.ndarray]:
        results = []
        if len(boxes) == 0:
            return [np.empty((0, 6)) for _ in range(self.cls_num)]
        for cls_id in range(self.cls_num):
            cls_inds = np.where(boxes[:,4] == cls_id)[0]
            results.append(boxes[cls_inds])
        return results

    def NMS_sort(
            self,
            boxes: np.ndarray,
            iou_thres: float
    ) -> np.ndarray:
        if len(boxes) == 0:
            return boxes
        
        x1s = boxes[:,0]
        y1s = boxes[:,1]
        # ws = boxes[:,2]
        # hs = boxes[:,3]
        # x2s = x1s + ws
        # y2s = y1s + hs
        x2s = boxes[:,2]
        y2s = boxes[:,3]
        ws = x2s - x1s
        hs = y2s - y1s
        areas = ws * hs

        order = np.argsort(boxes[:,5], axis=0)[::-1]
        results = []
        while order.size > 0:
            i = order[0]
            left = order[1:]
            results.append(i)

            xx1s = np.maximum(x1s[i], x1s[left])
            yy1s = np.maximum(y1s[i], y1s[left])
            xx2s = np.minimum(x2s[i], x2s[left])
            yy2s = np.minimum(y2s[i], y2s[left])
            w = np.maximum(0, xx2s - xx1s + 1e-5)
            h = np.maximum(0, yy2s - yy1s + 1e-5)
            intersect = w * h
            union = areas[i] + areas[left] - intersect
            iou = intersect / (union + 1e-6)
            
            slt_inds = np.where(iou <= iou_thres)[0]
            order = order[slt_inds + 1]
        
        return boxes[results]

    def postprocess(
        self,
        batch_num: int = 1,
        conf_thres: float = 0.75,
        iou_thres: float = 0.75
    ) -> List[List[float]]:
        
        output_batches = self.output_tensor.asnumpy()[:batch_num]
        output_batches = self.convert_centre_to_corner(output_batches) #中心坐标变换四角坐标
            
        all_results = []
        for batch in output_batches:
            boxes = self.specify(batch, conf_thres)
            clsed_boxes = self.classify(boxes)
            nmsed_clsed_boxes = []
            for cls_boxes in clsed_boxes:
                nmsed_boxes = self.NMS_sort(cls_boxes, iou_thres)
                nmsed_clsed_boxes.append(nmsed_boxes.tolist())
            all_results.append(nmsed_clsed_boxes)

        return all_results

    @abstractmethod
    def __call__(self):
        pass

class VisualDetect(YoloDetect):
    def __call__(
            self,
            imgs: Union[List[str], List[np.ndarray], List[sail.BMImage]],
            conf_thres: float = 0.75,
            iou_thres: float = 0.5
    ) -> List[List]:
        img_num = len(imgs)
        all_batches_inds = [list(range(i, min(i + self.batch_size, img_num))) for i in range(0, img_num, self.batch_size)]

        all_results = []
        for batches_inds in all_batches_inds:
            batch_imgs = [imgs[i] for i in batches_inds]

            transform_infos = self.preprocess(imgs=batch_imgs)
            
            self.net.process(
                self.graph_name,
                self.input_tensors,
                self.input_shapes,
                self.output_tensors)
            
            results = self.postprocess(
                batch_num=len(imgs),
                conf_thres=conf_thres,
                iou_thres=iou_thres
            )

            for i in range(len(batch_imgs)):
                results[i].append(transform_infos[i])

            #all_results.append(results)
            all_results.extend(results)

        return all_results

class VisualRecognize:
    def __init__(
        self,
        bmodel_rec: str,
        char_dict_path: str,
        dev_id: int = 0,
        img_size: List[List] = [[640, 48],[320, 48]],
        use_space_char: bool = True, #识别空格
        use_beam_search: bool = False, #使用启发式搜索
        beam_size: int = 5 #启发式搜索尺寸
    ):
        self.bmodel_rec = bmodel_rec
        self.dev_id = dev_id
        self.net = sail.Engine(
            self.bmodel_rec,
            self.dev_id,
            sail.IOMode.SYSIO
        )

        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)

        self.rec_batch_size = self.input_shape[0] # Max batch size in model stages.

        self.img_size = img_size
        self.img_size = sorted(self.img_size, key=lambda x: x[0])
        self.img_ratio = [x[0]/x[1] for x in self.img_size]
        self.img_ratio = sorted(self.img_ratio)

        # 解析字符字典
        self.character = ['blank']
        self.char_dict_path = char_dict_path
        try:
            with open(self.char_dict_path, "r", encoding="utf-8") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.strip("\n").strip("\r\n")
                    self.character.append(line)
        except FileExistsError:
            raise FileExistsError(f"在{self.char_dict_path}下没有找到字符字典。")
        except json.JSONDecodeError:
            raise ValueError("字符字典解析失败。")
        
        if use_space_char:
            self.character.append(" ")
        self.beam_search = use_beam_search
        self.beam_size = beam_size

        logging.info("模型已成功装载进入TPU设备内存。")
    
    def __del__(self):
        logging.info("模型已成功从TPU设备内存中卸载。")
    
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        h, w, _ = img.shape
        ratio = w / float(h)
        if ratio > self.img_ratio[-1]:
            logging.debug("Warning: ratio out of range: h = %d, w = %d, ratio = %f, bmodel with larger width is recommended."%(h, w, ratio))
            resized_w = self.img_size[-1][0]
            resized_h = self.img_size[-1][1]
            padding_w = resized_w
        else:          
            for max_ratio in self.img_ratio:
                if ratio <= max_ratio:
                    resized_h = self.img_size[0][1]
                    resized_w = int(resized_h * ratio)
                    padding_w = int(resized_h * max_ratio)
                    break
            
        if h != resized_h or w != resized_w:
            img = cv2.resize(img, (resized_w, resized_h))
        img = img.astype('float32')
        img = np.transpose(img, (2, 0, 1))
        img -= 127.5
        img *= 0.0078125

        padding_im = np.zeros((3, resized_h, padding_w), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = img
        
        return padding_im

    def predict(self, tensor: np.ndarray) -> np.ndarray:
        input_data = {self.input_name: np.array(tensor, dtype=np.float32)}
        outputs = self.net.process(self.graph_name, input_data)
        return list(outputs.values())[0]

    def postprocess(self, outputs: np.ndarray) -> List[str]:
        result_list = []

        if self.beam_search:
            max_seq_len, num_classes = outputs.shape[1], outputs.shape[2]

            for batch_idx in range(outputs.shape[0]):
                beams = [{'prefix': [], 'score': 1.0, 'confs':[]}] 

                for t in range(max_seq_len):
                    new_beams = []

                    for beam in beams:

                        next_char_probs = outputs[batch_idx, t]
                        top_candidates = np.argsort(-next_char_probs)[:self.beam_width]

                        for c in top_candidates:
                                new_prefix = beam['prefix'] + [c]
                                new_score = beam['score'] * next_char_probs[c]
                                new_confs = beam['confs'] + [next_char_probs[c]]
                                new_beams.append({'prefix': new_prefix, 'score': new_score, 'confs':new_confs})

                    new_beams.sort(key=lambda x: -x['score'])
                    beams = new_beams[:self.beam_width]

                best_beam = max(beams, key=lambda x: x['score'])

                char_list = []
                conf_list = []
                pre_c = best_beam['prefix'][0]
                if pre_c != 0:
                    char_list.append(self.character[pre_c])
                    conf_list.append(best_beam['confs'][0])
                for idx, c in enumerate(best_beam['prefix']):
                    if (pre_c==c) or (c==0):
                        if c ==0:
                            pre_c = c
                        continue
                    char_list.append(self.character[c])
                    conf_list.append(best_beam['confs'][idx])
                    pre_c = c
                result_list.append((''.join(char_list), np.mean(conf_list)))

        else:  # original postprocess
            preds_idx = outputs.argmax(axis=2)
            preds_prob = outputs.max(axis=2)
            for batch_idx, pred_idx in enumerate(preds_idx):
                char_list = []
                conf_list = []
                pre_c = pred_idx[0]
                if pre_c != 0:
                    char_list.append(self.character[pre_c])
                    conf_list.append(preds_prob[batch_idx][0])
                for idx, c in enumerate(pred_idx):
                    if (pre_c == c) or (c == 0):
                        if c == 0:
                            pre_c = c
                        continue
                    char_list.append(self.character[c])
                    conf_list.append(preds_prob[batch_idx][idx])
                    pre_c = c

                result_list.append((''.join(char_list), float(np.mean(conf_list))))

        return result_list

    def __call__(self, img_list: List[np.ndarray]) -> List[tuple[str, np.float32]]:
        img_dict = {}
        for img_size in self.img_size:
            img_dict[img_size[0]] = {"imgs":[], "ids":[], "res":[]}
        for id, img in enumerate(img_list):
            img = self.preprocess(img)
            if img is None:
                continue
            img_dict[img.shape[2]]["imgs"].append(img)
            img_dict[img.shape[2]]["ids"].append(id)

        for size_w in img_dict.keys():
            if size_w > 640:
                for img_input in img_dict[size_w]["imgs"]:
                    img_input = np.expand_dims(img_input, axis=0)
                    outputs = self.predict(img_input)
                    res = self.postprocess(outputs)
                    img_dict[size_w]["res"].extend(res)
            else:
                img_num = len(img_dict[size_w]["imgs"])
                for beg_img_no in range(0, img_num, self.rec_batch_size):
                    end_img_no = min(img_num, beg_img_no + self.rec_batch_size)
                    if beg_img_no + self.rec_batch_size > img_num:
                        for ino in range(beg_img_no, end_img_no):
                            img_input = np.expand_dims(img_dict[size_w]["imgs"][ino], axis=0)
                            outputs = self.predict(img_input)
                            res = self.postprocess(outputs)
                            img_dict[size_w]["res"].extend(res)   
                    else:
                        img_input = np.stack(img_dict[size_w]["imgs"][beg_img_no:end_img_no])
                        outputs = self.predict(img_input)
                        res = self.postprocess(outputs)
                        img_dict[size_w]["res"].extend(res)

        rec_res = []
        for size_w in img_dict.keys():
            rec_res.extend(img_dict[size_w]["res"])
        return rec_res



# 台阶代码底层函数
# basic_detects.py 文件的新增內容

import numpy as np
import sophon.sail as sail
import logging
from typing import Union, List



# ... 你原有的 YoloDetect, VisualDetect, VisualRecognize 類保持不變 ...


# =========================================================================
# ================ 專為台階檢測模型定製的新 class =========================
# =========================================================================

class StairDetect:
    """
    一个专门用于处理特殊多输出台阶检测模型的类。
    【最终修正版】：使用了 "临时文件 + sail.Decoder" 的方式来兼容旧版SDK。
    """
    def __init__(
        self,
        bmodel_path: str,
        dev_id: int = 0,
        conf_thresh: float = 0.5,
        nms_thresh: float = 0.5
    ):
        # ... __init__ 方法中的所有代码【保持不变】，这里省略以保持简洁 ...
        # (和上一版完全相同，从 self.net = sail.Engine(...) 开始)
        # 1. 初始化引擎和基本句柄
        self.net = sail.Engine(bmodel_path, dev_id, sail.IOMode.SYSO)
        self.handle = sail.Handle(dev_id)
        self.bmcv = sail.Bmcv(self.handle)
        self.graph_name = self.net.get_graph_names()[0]
        # 2. 獲取模型輸入信息
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.input_dtype = self.net.get_input_dtype(self.graph_name, self.input_name)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
        self.input_scale = self.net.get_input_scale(self.graph_name, self.input_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.input_shapes = {self.input_name: self.input_shape}
        # 3. 獲取模型的多個輸出信息
        self.output_names = self.net.get_output_names(self.graph_name)
        self.output_tensors = {}
        for output_name in self.output_names:
            output_shape = self.net.get_output_shape(self.graph_name, output_name)
            output_dtype = self.net.get_output_dtype(self.graph_name, output_name)
            self.output_tensors[output_name] = sail.Tensor(self.handle, output_shape, output_dtype, True, True)
        # 4. 初始化模型相關參數
        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        # 5. 初始化專用的後處理 PostProcess
        self.postprocess = PostProcessDetect(
            conf_thresh=conf_thresh,
            nms_thresh=nms_thresh,
            agnostic=False,
            multi_label=False,
            max_det=300,
        )
        # 6. 初始化預處理參數
        self.ab = [x * self.input_scale / 255. for x in [1, 0, 1, 0, 1, 0]]
        logging.info(f"StairDetect model '{bmodel_path}' loaded successfully.")


    def _numpy_to_bmimage(self, np_img: np.ndarray) -> sail.BMImage:
        """
        【新增】一个内部辅助函数，使用 "临时文件" 方法将Numpy数组转换为BMImage。
        此逻辑借鉴自您最初的 YoloDetect.convert_to_bmimage 方法。
        """
        tmp_file = None
        try:
            # 创建一个带唯一名称的临时文件
            tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp_file_path = tmp_file.name
            tmp_file.close()

            # 将Numpy数组写入这个临时文件
            cv2.imwrite(tmp_file_path, np_img)
            
            # 使用 sail.Decoder 读取这个临时文件，这和例程的逻辑一致
            decoder = sail.Decoder(tmp_file_path, True, self.handle.get_device_id())
            bmimg = sail.BMImage()
            ret = decoder.read(self.handle, bmimg)
            if ret != 0:
                logging.error(f"Failed to decode temporary file: {tmp_file_path}")
                # 返回一个空的 BMImage 以避免后续崩溃
                return sail.BMImage()
            return bmimg
        finally:
            # 确保临时文件在使用后被删除
            if tmp_file and os.path.exists(tmp_file.name):
                os.remove(tmp_file.name)

    # 【注意】我们不再需要 _preprocess 方法，因为 __call__ 会直接处理
    # 之前版本的 _preprocess 方法可以整个删除

    def __call__(self, imgs: List[np.ndarray]) -> List[List]:
        """
        主调用函数，接口与 VisualDetect 保持一致。
        内部逻辑已完全重构，以匹配例程的工作方式。
        """
        img_num = len(imgs)
        if img_num > self.batch_size:
            raise ValueError(f"Input batch size {img_num} exceeds model's batch size {self.batch_size}.")

        bmimg_list = []
        for img in imgs:
            bmimg = self._numpy_to_bmimage(img)
            if bmimg.height() == 0: # <--- 使用这个更可靠的检查方法
                continue
            bmimg_list.append(bmimg)
        
        if not bmimg_list:
            logging.warning("All numpy images failed to convert to BMImage.")
            return []

        # --- 以下逻辑完全来自能够成功运行的官方例程 __call__ 方法 ---
        ori_size_list = []
        ratio_list = []
        txy_list = []
        transform_infos = []
        preprocessed_bmimgs = []
        
        # 1. 预处理 (来自例程 preprocess_bmcv)
        for bmimg in bmimg_list:
            ori_size_list.append((bmimg.width(), bmimg.height()))
            rgb_planar_img = sail.BMImage(self.handle, bmimg.height(), bmimg.width(), sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
            self.bmcv.convert_format(bmimg, rgb_planar_img)
            resized_img_rgb, ratio, txy = self.resize_bmcv(rgb_planar_img) # 调用例程的resize方法
            preprocessed_bmimg = sail.BMImage(self.handle, self.net_h, self.net_w, sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
            self.bmcv.convert_to(resized_img_rgb, preprocessed_bmimg, ((self.ab[0], self.ab[1]), (self.ab[2], self.ab[3]), (self.ab[4], self.ab[5])))
            preprocessed_bmimgs.append(preprocessed_bmimg)
            ratio_list.append(ratio)
            txy_list.append(txy)
            transform_infos.append({"org_size": (bmimg.width(), bmimg.height()), "ratio": ratio, "txy": txy})
            
        # 2. 推理 (来自例程 predict)
        if img_num > 1:
            BMImageArray = getattr(sail, f'BMImageArray{self.batch_size}D')
            bmimg_array = BMImageArray()
            for i in range(img_num):
                bmimg_array[i] = preprocessed_bmimgs[i].data()
            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype,  False, False)
            self.bmcv.bm_image_to_tensor(bmimg_array, input_tensor)
        else:
            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype,  False, False)
            self.bmcv.bm_image_to_tensor(preprocessed_bmimgs[0], input_tensor)

        outputs = self.predict(input_tensor, img_num)

        # 3. 后处理 (调用例程的 postprocess)
        results = self.postprocess(outputs, ori_size_list, ratio_list, txy_list)
        
        # 4. 格式化输出 (与你之前的接口保持一致)
        cls_num = 1 # 假设台阶模型只有1个类别
        return self._reformat_output(results, cls_num, transform_infos)

    # --- 以下是需要从例程YOLOv8类中完整复制过来的辅助方法 ---
    # 我们需要 resize_bmcv, predict, _reformat_output(之前已提供)
    
    def resize_bmcv(self, bmimg):
        img_w = bmimg.width()
        img_h = bmimg.height()
        r_w = self.net_w / img_w
        r_h = self.net_h / img_h
        r = min(r_w, r_h)
        tw = int(round(r * img_w))
        th = int(round(r * img_h))
        tx1, ty1 = self.net_w - tw, self.net_h - th
        tx1 /= 2
        ty1 /= 2
        ratio = (r, r)
        txy = (tx1, ty1)
        attr = sail.PaddingAtrr()
        attr.set_stx(int(round(tx1 - 0.1)))
        attr.set_sty(int(round(ty1 - 0.1)))
        attr.set_w(tw)
        attr.set_h(th)
        attr.set_r(114)
        attr.set_g(114)
        attr.set_b(114)
        resized_img_rgb = self.bmcv.crop_and_resize_padding(bmimg, 0, 0, img_w, img_h, self.net_w, self.net_h, attr, sail.bmcv_resize_algorithm.BMCV_INTER_LINEAR)
        return resized_img_rgb, ratio, txy

    def predict(self, input_tensor, img_num):
        input_tensors = {self.input_name: input_tensor} 
        self.net.process(self.graph_name, input_tensors, self.input_shapes, self.output_tensors)
        outputs_dict = {}
        for name in self.output_names:
            outputs_dict[name] = self.output_tensors[name].asnumpy()[:img_num].transpose(0,2,1)
        out_keys = list(outputs_dict.keys())
        ord = []
        for n in self.output_names:
            for i, k in enumerate(out_keys):
                if n in k:
                    ord.append(i)
                    break
        out = [outputs_dict[out_keys[i]] for i in ord]
        return out

    def _reformat_output(self, dets_for_all_imgs: List[np.ndarray], cls_num: int, transform_infos: List[dict]) -> List[List]:
        reformatted_results = []
        for i, dets in enumerate(dets_for_all_imgs):
            class_results = [[] for _ in range(cls_num)]
            for det in dets:
                box_and_score = det[:5].tolist()
                class_id = int(det[5])
                if class_id < cls_num:
                    class_results[class_id].append(box_and_score)
            for j in range(cls_num):
                class_results[j] = np.array(class_results[j])
            class_results.append(transform_infos[i])
            reformatted_results.append(class_results)
        return reformatted_results

    
# =========================================================================
# ============ 专为 YOLOv8 分割模型(Seg)定製的新 class ====================
# =========================================================================

class YoloSegDetect:
    """
    一个专门用于处理 YOLOv8 分割模型的类。
    它整合了官方例程的核心逻辑，并封装了与您现有框架一致的接口，
    其输出可以直接被 `generate_locations` 函数使用。
    """
    def __init__(
        self,
        bmodel_path: str,
        dev_id: int = 0,
        conf_thresh: float = 0.25,
        nms_thresh: float = 0.7
    ):
        # 1. 初始化引擎和基本句柄 (来自例程 __init__)
        self.net = sail.Engine(bmodel_path, dev_id, sail.IOMode.SYSO)
        self.handle = sail.Handle(dev_id)
        self.bmcv = sail.Bmcv(self.handle)
        self.graph_name = self.net.get_graph_names()[0]
        
        # 2. 获取模型输入信息
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.input_dtype = self.net.get_input_dtype(self.graph_name, self.input_name)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
        self.input_scale = self.net.get_input_scale(self.graph_name, self.input_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.input_shapes = {self.input_name: self.input_shape}

        # 3. 获取模型的多个输出信息
        self.output_names = self.net.get_output_names(self.graph_name)
        self.output_tensors = {}
        for output_name in self.output_names:
            output_shape = self.net.get_output_shape(self.graph_name, output_name)
            output_dtype = self.net.get_output_dtype(self.graph_name, output_name)
            self.output_tensors[output_name] = sail.Tensor(self.handle, output_shape, output_dtype, True, True)

        # 4. 初始化模型相关参数
        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        
        # 5. 初始化专用的后处理 PostProcess (来自分割例程)
        self.postprocess = PostProcessSeg(
            conf_thres=conf_thresh,
            iou_thres=nms_thresh
        )
        
        # 6. 初始化预处理参数
        self.ab = [x * self.input_scale / 255. for x in [1, 0, 1, 0, 1, 0]]
        
        logging.info(f"YoloSegDetect model '{bmodel_path}' loaded successfully.")

    def _numpy_to_bmimage(self, np_img: np.ndarray) -> sail.BMImage:
        # 这个函数使用 "临时文件" 方法，已在您的环境中验证可用
        tmp_file = None
        try:
            tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp_file_path = tmp_file.name
            tmp_file.close()
            cv2.imwrite(tmp_file_path, np_img)
            decoder = sail.Decoder(tmp_file_path, True, self.handle.get_device_id())
            bmimg = sail.BMImage()
            if decoder.read(self.handle, bmimg) != 0:
                logging.error(f"Failed to decode temporary file: {tmp_file_path}")
                return sail.BMImage() # 返回一个无效对象
            return bmimg
        finally:
            if tmp_file and os.path.exists(tmp_file.name):
                os.remove(tmp_file.name)

    def _reformat_output_for_locations(self, results_from_model: list, transform_infos: list, cls_num: int = 80) -> list:
        """
        【核心转换】将分割模型的输出，格式化为 `generate_locations` 函数需要的格式。
        输入: results_from_model 是一个列表，每个元素是元组 (boxes, segments, masks)
        输出: 统一接口格式 [[cls0_boxes], [cls1_boxes], ..., transform_info]
        """
        reformatted_results = []
        for i, result_tuple in enumerate(results_from_model):
            boxes, segments, masks = result_tuple
            
            # 创建一个空的列表来存放每个类别的结果
            class_results = [[] for _ in range(cls_num)]
            
            # boxes 的格式是 (N, 6)，列为 x1, y1, x2, y2, score, class_id
            for det in boxes:
                # generate_locations 需要 x1, y1, x2, y2
                box_coords = det[:4].tolist() 
                class_id = int(det[5])
                if class_id < cls_num:
                    class_results[class_id].append(box_coords)
            
            # 将内部的 list 转换为 numpy array
            for j in range(cls_num):
                class_results[j] = np.array(class_results[j])
                
            # 在末尾添加 transform_info
            class_results.append(transform_infos[i])
            reformatted_results.append(class_results)
            
        return reformatted_results

    def __call__(self, imgs: List[np.ndarray]) -> List[List]:
        """
        主调用函数，接口与 VisualDetect/StairDetect 保持一致。
        """
        img_num = len(imgs)
        if img_num > self.batch_size:
            raise ValueError(f"Input batch size {img_num} exceeds model's batch size {self.batch_size}.")

        bmimg_list = [self._numpy_to_bmimage(img) for img in imgs]
        bmimg_list = [bmimg for bmimg in bmimg_list if bmimg.height() > 0]
        
        if not bmimg_list:
            logging.warning("All numpy images failed to convert to BMImage.")
            return []
        
        actual_img_num = len(bmimg_list)
        ori_size_list, ratio_list, txy_list, transform_infos = [], [], [], []
        
        # 1. 预处理
        preprocessed_bmimgs = []
        for bmimg in bmimg_list:
            ori_size_list.append((bmimg.width(), bmimg.height()))
            rgb_planar_img = sail.BMImage(self.handle, bmimg.height(), bmimg.width(), sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
            self.bmcv.convert_format(bmimg, rgb_planar_img)
            resized_img_rgb, ratio, txy = self.resize_bmcv(rgb_planar_img)
            preprocessed_bmimg = sail.BMImage(self.handle, self.net_h, self.net_w, sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
            self.bmcv.convert_to(resized_img_rgb, preprocessed_bmimg, ((self.ab[0], self.ab[1]), (self.ab[2], self.ab[3]), (self.ab[4], self.ab[5])))
            
            preprocessed_bmimgs.append(preprocessed_bmimg)
            ratio_list.append(ratio)
            txy_list.append(txy)
            transform_infos.append({"org_size": (bmimg.width(), bmimg.height()), "ratio": ratio, "txy": txy})

        # 2. 推理
        if actual_img_num > 1:
            BMImageArray = getattr(sail, f'BMImageArray{self.batch_size}D')
            bmimg_array = BMImageArray()
            for i in range(actual_img_num):
                bmimg_array[i] = preprocessed_bmimgs[i].data()
            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype, False, False)
            self.bmcv.bm_image_to_tensor(bmimg_array, input_tensor)
        else:
            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype, False, False)
            self.bmcv.bm_image_to_tensor(preprocessed_bmimgs[0], input_tensor)

        outputs = self.predict(input_tensor, actual_img_num)
        
        # 3. 后处理 (调用分割模型专用的 postprocess)
        results = self.postprocess(outputs, ori_size_list, ratio_list, txy_list)
        
        # 4. 格式化输出以匹配 generate_locations 函数的需要
        # return self._reformat_output_for_locations(results, transform_infos)
        return results

    # --- 以下是需要从Yolov8Seg例程中完整复制过来的辅助方法 ---
    def resize_bmcv(self, bmimg):
        img_w, img_h = bmimg.width(), bmimg.height()
        r = min(self.net_w / img_w, self.net_h / img_h)
        tw, th = int(round(r * img_w)), int(round(r * img_h))
        tx1, ty1 = (self.net_w - tw) / 2, (self.net_h - th) / 2
        ratio, txy = (r, r), (tx1, ty1)
        attr = sail.PaddingAtrr()
        attr.set_stx(int(round(tx1 - 0.1))); attr.set_sty(int(round(ty1 - 0.1)))
        attr.set_w(tw); attr.set_h(th)
        attr.set_r(114); attr.set_g(114); attr.set_b(114)
        resized_img_rgb = self.bmcv.crop_and_resize_padding(bmimg, 0, 0, img_w, img_h, self.net_w, self.net_h, attr, sail.bmcv_resize_algorithm.BMCV_INTER_LINEAR)
        return resized_img_rgb, ratio, txy

    def predict(self, input_tensor, img_num):
        input_tensors = {self.input_name: input_tensor} 
        self.net.process(self.graph_name, input_tensors, self.input_shapes, self.output_tensors)
        outputs_dict = {}
        for name in self.output_names:
            # 注意：此处的 predict 与之前的 StairDetect 不同，没有 transpose()
            outputs_dict[name] = self.output_tensors[name].asnumpy()[:img_num]
        out_keys = list(outputs_dict.keys())
        ord = [out_keys.index(n) for n in self.output_names if n in out_keys]
        return [outputs_dict[out_keys[i]] for i in ord]