# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

import shutil
import json

import ultralytics
import time


print(f"PyTorch version: {torch.__version__}")
print(ultralytics.__version__)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):

    # project_directory
    _output_directory = os.path.dirname(os.path.abspath(__file__))

    conf_file = os.path.join(_output_directory, "detect.conf") #set 1 to loop runing
    if not os.path.exists(conf_file):
        with open(conf_file + '.txt', 'a') as f:
            f.write('1')

    input_img_path = os.path.join(_output_directory, "detect_input")
    if os.path.exists(input_img_path):
        shutil.rmtree(input_img_path)
    os.makedirs(input_img_path)

    output_img_path = os.path.join(_output_directory, "detect_out")
    if os.path.exists(output_img_path):
        shutil.rmtree(output_img_path)
    os.mkdir(output_img_path)

    output_img_crop_path = os.path.join(_output_directory, "detect_out_crop")
    if os.path.exists(output_img_crop_path):
        shutil.rmtree(output_img_crop_path)
    os.mkdir(output_img_crop_path)

    output_img_crop_label_path = os.path.join(_output_directory, "detect_out_crop_label")
    if os.path.exists(output_img_crop_label_path):
        shutil.rmtree(output_img_crop_label_path)
    os.mkdir(output_img_crop_label_path)

    _times = []

    _output_directory = os.path.dirname(os.path.abspath(__file__))
    _file_exec_time = os.path.join(_output_directory, "exec_time.json")
    try:
        os.remove(_file_exec_time)
        print(f"File {_file_exec_time} has been deleted.")
    except FileNotFoundError:
        print(f"File {_file_exec_time} does not exist.")
    except PermissionError:
        print(f"Permission denied: Unable to delete {_file_exec_time}.")
    except Exception as e:
        print(f"An error occurred: {e}")

    with open(_file_exec_time, "w") as filex:
        json.dump([], filex, indent=4)  # Use 'indent' for pretty formatting




    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    print(save_dir)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    while True:
        with open(conf_file, 'r') as f:
            status = f.read()
        if status != "1":
            break
        for r, d, f in os.walk(input_img_path):
            if len(f)>0:
                print("file in coming")
            for file in f:
                # print(file)

                result_preds = []
                extension_allowed = '.jpg'
                ext_len = len(extension_allowed)
                strip = file[0:len(file) - ext_len]
                if os.path.exists(os.path.join(output_img_crop_path, strip)):
                    shutil.rmtree(os.path.join(output_img_crop_path, strip))
                os.mkdir(os.path.join(output_img_crop_path, strip))
                if os.path.exists(os.path.join(output_img_crop_label_path, strip)):
                    shutil.rmtree(os.path.join(output_img_crop_label_path, strip))
                os.mkdir(os.path.join(output_img_crop_label_path, strip))
                i_crop = 0
                source = os.path.join(input_img_path, file)

                source = str(source)
                save_img = not nosave and not source.endswith('.txt')  # save inference images
                is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
                is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
                webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
                if is_url and is_file:
                    source = check_file(source)  # download

                print("source",source)

                # Half
                half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
                if pt or jit:
                    model.model.half() if half else model.model.float()

                # start excution time
                start_time = time.time()


                # Dataloader
                if webcam:
                    view_img = check_imshow()
                    cudnn.benchmark = True  # set True to speed up constant image size inference
                    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
                    bs = len(dataset)  # batch_size
                else:
                    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
                    bs = 1  # batch_size
                vid_path, vid_writer = [None] * bs, [None] * bs

                # Run inference
                model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
                dt, seen = [0.0, 0.0, 0.0], 0
                for path, im, im0s, vid_cap, s in dataset:
                    t1 = time_sync()
                    im = torch.from_numpy(im).to(device)
                    im = im.half() if half else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim
                    t2 = time_sync()
                    dt[0] += t2 - t1

                    # Inference
                    visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                    pred = model(im, augment=augment, visualize=visualize)
                    t3 = time_sync()
                    dt[1] += t3 - t2

                    # NMS
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                    dt[2] += time_sync() - t3

                    # Second-stage classifier (optional)
                    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

                    # Process predictions
                    for i, det in enumerate(pred):  # per image
                        seen += 1
                        if webcam:  # batch_size >= 1
                            p, im0, frame = path[i], im0s[i].copy(), dataset.count
                            s += f'{i}: '
                        else:
                            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                        p = Path(p)  # to Path
                        save_path = str(save_dir / p.name)  # im.jpg
                        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                        s += '%gx%g ' % im.shape[2:]  # print string
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        imc = im0.copy() if save_crop else im0  # for save_crop
                        annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                            # Print results
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string


                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                if save_txt:  # Write to file
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                    with open(txt_path + '.txt', 'a') as f:
                                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                                if save_img or save_crop or view_img:  # Add bbox to image
                                    c = int(cls)  # integer class
                                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                    annotator.box_label(xyxy, label, color=colors(c, True))
                                    if save_crop:
                                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                                    else:

                                        from utils.general import (clip_coords, xywh2xyxy, xyxy2xywh)

                                        # print("xywh1", xyxy)
                                        xyxy = torch.tensor(xyxy).view(-1, 4)

                                        b = xyxy2xywh(xyxy)  # boxes
                                        gain = 1.02
                                        pad = 10
                                        b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
                                        xyxy = xywh2xyxy(b).long()
                                        print("xywh", xyxy)
                                        clip_coords(xyxy, im.shape)
                                        print("b", b)
                                        print("conf", str(conf.tolist()))



                                        bbs = get_print_bb(str(b))
                                        print("bbs", bbs)
                                        xywh_result = crop_img(os.path.join(input_img_path, file), bbs,
                                                 os.path.join(output_img_crop_path, strip, str(i_crop) + ".jpg"))

                                        result_pred = {}
                                        result_pred["label_id"] = int(c)
                                        result_pred["label"] = names[int(c)]
                                        result_pred["xywh"] = xywh_result
                                        result_pred["conf"] = str(conf.tolist())
                                        result_preds.append(result_pred)
                                        with open(os.path.join(output_img_crop_label_path, strip, str(i_crop) + ".json"), "w") as outfile:
                                            outfile.write(json.dumps(result_pred))

                                        i_crop = i_crop + 1



                        # Print time (inference-only)
                        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

                        # Stream results
                        im0 = annotator.result()
                        if view_img:
                            cv2.imshow(str(p), im0)
                            cv2.waitKey(1)  # 1 millisecond

                        # Save results (image with detections)
                        if save_img:
                            if dataset.mode == 'image':
                                # cv2.imwrite(save_path, im0)
                                cv2.imwrite( os.path.join(output_img_path, file), im0 )
                            else:  # 'video' or 'stream'
                                if vid_path[i] != save_path:  # new video
                                    vid_path[i] = save_path
                                    if isinstance(vid_writer[i], cv2.VideoWriter):
                                        vid_writer[i].release()  # release previous video writer
                                    if vid_cap:  # video
                                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                    else:  # stream
                                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                                        save_path += '.mp4'
                                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                                vid_writer[i].write(im0)

                    print("\n")



                    # stop excution time
                    end_time = time.time()
                    execution_time = end_time - start_time

                    # if os.path.exists(_file_exec_time):
                    #     with open(_file_exec_time, "r") as file:
                    #         _times = json.load(file)

                    _times.append(execution_time)
                    # print(_times)
                    print("execute: "+_file_exec_time)
                    with open(_file_exec_time, "w") as filex:
                        json.dump(_times, filex, indent=4)  # Use 'indent' for pretty formatting




                # Print results
                t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
                LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
                if save_txt or save_img:
                    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
                if update:
                    strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

                print(os.path.join(input_img_path, file))
                os.remove( os.path.join(input_img_path, file) )

def crop_img(img_path, bbs, out): #bbs = x, y, w, h
    xywh = []
    i = 0
    for bb in bbs:
        x = int(bb[0][0]-(bb[0][2]/2))
        if x < 0:
            x = 0
        y = int(bb[0][1]-(bb[0][3]/2))
        if y < 0:
            y = 0
        w = int(bb[0][2])
        h = int(bb[0][3])
        xywh.append(x)
        xywh.append(y)
        xywh.append(w)
        xywh.append(h)
        # print(img_path)
        img = cv2.imread(img_path)
        imgCrop = img[y:y + h, x:x + w]

        print("crop ",out)
        cv2.imwrite(out, imgCrop)
        i = i + 1
    return xywh

def get_print_bb(bb_str):
    ds = bb_str.split("tensor(")
    # print(ds)
    bb_list = []
    if len(ds)>1:
        i = 0
        for data in ds:
            if(i != 0):
                bb = data.split(")")[0]
                bb1 = bb.split(".,")
                bb = ""
                for a in range(len(bb1)):
                    bb = bb + bb1[a]
                    if(a < len(bb1)-1):
                        bb = bb + ","
                bb = bb.split(".]]")[0].split("]]")[0]+"]]"
                print("----",bb)
                bb_list.append(json.loads(bb))
            i=i+1
        return bb_list
    else:
        return bb_list

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)






