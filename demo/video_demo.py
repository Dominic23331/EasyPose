import argparse

import cv2
import easypose as ep


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', help='video file.')

    parser.add_argument('--pose-model', type=str, default='rtmpose_tiny', help='pose model name.')
    parser.add_argument('--decoder', type=str, default='SimCC', help='pose model decoder.')
    parser.add_argument('--det-model', type=str, default='rtmdet_tiny', help='detection model.')

    parser.add_argument('--conf', type=float, default=0.6, help='det model confidence threshold.')
    parser.add_argument('--iou', type=float, default=0.6, help='det model iou threshold.')

    parser.add_argument('--device', type=str, default='CUDA', help='device used for inference.')
    parser.add_argument('--warmup', type=int, default=30, help='warmup epoch.')

    parser.add_argument('--show', type=bool, default=True, help='display video.')
    parser.add_argument('--save', type=str, default='', help='save video.')

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

    cap = cv2.VideoCapture(args.video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        raise ValueError("Unable to open video file: {}".format(args.video_path))

    if args.save:
        # 获取视频帧的宽度和高度
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        # 设置保存视频的编解码器和输出文件
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args.save, fourcc, 30.0, (frame_width, frame_height))

    while cap.isOpened():
        # 读取一帧
        ret, frame = cap.read()

        if ret:
            poses = model.predict(frame)

            frame = ep.draw_keypoints(frame,
                                      poses,
                                      radius=args.radius,
                                      thickness=args.thickness,
                                      box_thickness=args.box_thickness,
                                      draw_box=args.draw_box)

            if args.show:
                # 在窗口中显示视频
                cv2.imshow('Video', frame)
            if args.save:
                # 将帧写入输出文件
                out.write(frame)

            # 检查是否按下'q'键，如果是则退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # 释放资源
    cap.release()
    if args.save:
        out.release()
    if args.show:
        # 关闭窗口
        cv2.destroyAllWindows()
