import argparse

import cv2
import easypose as ep


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('img', help='image file.')

    parser.add_argument('--pose-model', type=str, default='rtmpose_tiny', help='pose model name.')
    parser.add_argument('--decoder', type=str, default='SimCC', help='pose model decoder.')
    parser.add_argument('--det-model', type=str, default='rtmdet_tiny', help='detection model.')

    parser.add_argument('--conf', type=float, default=0.6, help='det model confidence threshold.')
    parser.add_argument('--iou', type=float, default=0.6, help='det model iou threshold.')

    parser.add_argument('--device', type=str, default='CUDA', help='device used for inference.')
    parser.add_argument('--warmup', type=int, default=30, help='warmup epoch.')

    parser.add_argument('--show', type=bool, default=True, help='display image.')
    parser.add_argument('--save', type=str, default='', help='save image.')

    parser.add_argument('--radius', type=int, default=5, help='radius of keypoints.')
    parser.add_argument('--thickness', type=int, default=-1, help='thickness of keypoints')
    parser.add_argument('--draw-box', type=bool, default=False, help='draw detection boxes.')
    parser.add_argument('--box-thickness', type=int, default=5, help='thickness of boxes.')

    args = parser.parse_args()

    if not args.show and not args.save:
        raise ValueError('The video must be displayed or saved.')

    return args


if __name__ == '__main__':
    args = get_args()

    model = ep.TopDown(pose_model_name=args.pose_model,
                       pose_model_decoder=args.decoder,
                       det_model_name=args.det_model,
                       conf_threshold=args.conf,
                       iou_threshold=args.iou,
                       device=args.device,
                       warmup=args.warmup)

    image = cv2.imread(args.img)

    poses = model.predict(image)

    image = ep.draw_keypoints(image,
                              poses,
                              radius=args.radius,
                              thickness=args.thickness,
                              box_thickness=args.box_thickness,
                              draw_box=args.draw_box)

    if args.show:
        cv2.imshow("result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if args.save:
        cv2.imwrite(args.save, image)
