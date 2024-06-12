import os, sys
import argparse
import gradio as gr
from pathlib import Path
from inference.real3d_infer import GeneFace2Infer


class Inferer(GeneFace2Infer):
    def infer_once_args(self, *args, **kargs):
        assert len(kargs) == 0
        keys = [
            "src_image_name",
            "drv_audio_name",
            "drv_pose_name",
            "bg_image_name",
            "blink_mode",
            "temperature",
            "mouth_amp",
            "out_mode",
            "SR_mode",
            "a2m_ckpt",
            "torso_ckpt",
            "Real_ESRGAN_ckpt",
            "GFPGAN_ckpt",
            "min_face_area_percent",
        ]
        inp = {}
        info = ""
        out_name = None

        try:
            for key_index in range(len(keys)):
                key = keys[key_index]
                inp[key] = args[key_index]
                if "_name" in key:
                    inp[key] = inp[key] if inp[key] is not None else ""

            if inp["src_image_name"] == "":
                info = "Input Error: 需要源图片"
                raise ValueError
            if inp["drv_audio_name"] == "" and inp["drv_pose_name"] == "":
                info = "Input Error: 至少需要一个驱动音频或视频"
                raise ValueError

            if inp["drv_audio_name"] == "" and inp["drv_pose_name"] != "":
                inp["drv_audio_name"] = inp["drv_pose_name"]
                print("无驱动音频, 使用姿势视频作为音频输入")
            if inp["drv_pose_name"] == "":
                inp["drv_pose_name"] = "static"
                print("无驱动视频, 使用静态姿势")

            reload_flag = False
            if inp["a2m_ckpt"] != self.audio2secc_ckpt:
                print("检测到a2m_ckpt更改, 重新加载模型")
                reload_flag = True
            if inp["torso_ckpt"] != self.torso_model_ckpt:
                print("检测到torso_ckpt更改, 重新加载模型")
                reload_flag = True
            if inp["Real_ESRGAN_ckpt"] != self.Real_ESRGAN_ckpt:
                print("检测到Real_ESRGAN_ckpt更改, 重新加载模型")
                reload_flag = True
            if inp["GFPGAN_ckpt"] != self.GFPGAN_ckpt:
                print("检测到GFPGAN_ckpt更改, 重新加载模型")
                reload_flag = True

            inp["seed"] = 42
            for key in inp:
                if "name" in key and inp[key] != "":
                    print(f"{key} : {inp[key]}")

            try:
                if reload_flag:
                    self.__init__(inp["a2m_ckpt"], inp["torso_ckpt"], inp=inp, device=self.device)
            except Exception as e:
                info = f"Reload ERROR: {e}"
                raise ValueError

            try:
                out_name = self.infer_once(inp)
            except Exception as e:
                info = f"Inference ERROR: {e}"
                raise ValueError

        except Exception as e:
            if info == "":
                info = f"WebUI ERROR: {e}"

        if len(info) > 0:
            print(info)
            info_gr = gr.update(visible=True, value=info)
        else:
            info_gr = gr.update(visible=False, value=info)
        if out_name is not None and len(out_name) > 0 and os.path.exists(out_name):
            print(f"成功生成视频 {out_name}")
            video_gr = gr.update(visible=True, value=out_name)
        else:
            print(f"生成视频失败")
            video_gr = gr.update(visible=False, value=out_name)
        return info_gr, video_gr


def gr_demo(
    audio2secc_ckpt,
    torso_model_ckpt,
    Real_ESRGAN_ckpt,
    GFPGAN_ckpt,
    device="cuda",
):

    infer_obj = Inferer(
        audio2secc_ckpt=audio2secc_ckpt,
        torso_model_ckpt=torso_model_ckpt,
        Real_ESRGAN_ckpt=Real_ESRGAN_ckpt,
        GFPGAN_ckpt=GFPGAN_ckpt,
        device=device,
    )

    with gr.Blocks(analytics_enabled=False, theme=gr.themes.Base()) as demo:
        gr.Markdown("<h1> 计算机图形学PJ3 </h1>")

        with gr.Row():
            with gr.Column(variant="panel"):
                with gr.Tabs(elem_id="source_image"):
                    with gr.TabItem("源图片"):
                        with gr.Row():
                            src_image_name = gr.Image(
                                show_label=False,
                                type="filepath",
                                value="data/Macron.png",
                            )
                with gr.Tabs(elem_id="driven_audio"):
                    with gr.TabItem("音频"):
                        with gr.Column(variant="panel"):
                            drv_audio_name = gr.Audio(
                                show_label=False,
                                type="filepath",
                                value="data/Obama_5s.wav",
                            )
                with gr.Tabs(elem_id="driven_pose"):
                    with gr.TabItem("姿势视频"):
                        with gr.Column(variant="panel"):
                            drv_pose_name = gr.Video(
                                show_label=False,
                                value="data/May_5s.mp4",
                            )
                with gr.Tabs(elem_id="bg_image"):
                    with gr.TabItem("背景图片"):
                        with gr.Row():
                            bg_image_name = gr.Image(
                                show_label=False,
                                type="filepath",
                                value="data/bg_white_house.png",
                            )

            with gr.Column(variant="panel"):
                with gr.Tabs(elem_id="checkbox"):
                    with gr.TabItem("参数设置"):
                        with gr.Column(variant="panel"):
                            blink_mode = gr.Radio(
                                ["none", "period"], value="period", label="blink mode", info="是否周期性眨眼"
                            )
                            min_face_area_percent = gr.Slider(
                                minimum=0.15,
                                maximum=0.5,
                                step=0.01,
                                label="min_face_area_percent",
                                value=0.2,
                                info="图片的最小面部比例 (过小需要裁剪)",
                            )
                            temperature = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                step=0.025,
                                label="temperature",
                                value=1.0,
                                info="音频驱动幅度",
                            )
                            mouth_amp = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                step=0.025,
                                label="mouth amplitude",
                                value=0.45,
                                info="嘴部动作幅度",
                            )
                            out_mode = gr.Radio(
                                ["none", "depth"], value="none", label="out_mode", info="是否显示深度图"
                            )
                            SR_mode = gr.Radio(
                                ["none", "GFPGAN-V1.4"], value="GFPGAN-V1.4", label="SR_mode", info="是否使用超分辨率"
                            )
                            submit = gr.Button("Generate", elem_id="generate", variant="primary")

                    with gr.Tabs(elem_id="genearted_video"):
                        with gr.TabItem("生成视频"):
                            info_gr = gr.Textbox(label="Error", interactive=False, visible=False)
                            video_gr = gr.Video(show_label=False, format="mp4", visible=True)

            with gr.Column(variant="panel"):
                with gr.Tabs(elem_id="checkbox"):
                    with gr.TabItem("预训练模型参数"):
                        with gr.Column(variant="panel"):
                            audio2secc_ckpt = gr.FileExplorer(
                                glob="checkpoints/**/*.ckpt",
                                value=audio2secc_ckpt,
                                file_count="single",
                                label="audio2secc model ckpt",
                            )
                            torso_model_ckpt = gr.FileExplorer(
                                glob="checkpoints/**/*.ckpt",
                                value=torso_model_ckpt,
                                file_count="single",
                                label="torso model ckpt",
                            )
                            Real_ESRGAN_ckpt = gr.FileExplorer(
                                glob="checkpoints/**/*.pth",
                                value=Real_ESRGAN_ckpt,
                                file_count="single",
                                label="Real_ESRGAN model ckpt",
                            )
                            GFPGAN_ckpt = gr.FileExplorer(
                                glob="checkpoints/**/*.pth",
                                value=GFPGAN_ckpt,
                                file_count="single",
                                label="GFPGAN model ckpt",
                            )

        submit.click(
            fn=infer_obj.infer_once_args,
            inputs=[
                src_image_name,
                drv_audio_name,
                drv_pose_name,
                bg_image_name,
                blink_mode,
                temperature,
                mouth_amp,
                out_mode,
                SR_mode,
                audio2secc_ckpt,
                torso_model_ckpt,
                Real_ESRGAN_ckpt,
                GFPGAN_ckpt,
                min_face_area_percent,
            ],
            outputs=[
                info_gr,
                video_gr,
            ],
        )

    return demo


# fmt:off
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    a2m_ckpt = Path("checkpoints/240210_real3dportrait_orig/audio2secc_vae/model_ckpt_steps_400000.ckpt").resolve().__str__()
    torso_ckpt = Path("checkpoints/240210_real3dportrait_orig/secc2plane_torso_orig/model_ckpt_steps_100000.ckpt").resolve().__str__()
    Real_ESRGAN_ckpt = Path("checkpoints/gfpgan/realesr-general-x4v3.pth").resolve().__str__()
    GFPGAN_ckpt = Path("checkpoints/gfpgan/GFPGANv1.4.pth").resolve().__str__()
    parser.add_argument("--a2m_ckpt", type=str, default=a2m_ckpt)
    parser.add_argument("--torso_ckpt", type=str, default=torso_ckpt,)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--server", type=str, default="127.0.0.1")
    parser.add_argument("--share", action="store_true", dest="share")
    args = parser.parse_args()
    demo = gr_demo(
        audio2secc_ckpt=args.a2m_ckpt,
        torso_model_ckpt=args.torso_ckpt,
        Real_ESRGAN_ckpt=Real_ESRGAN_ckpt,
        GFPGAN_ckpt=GFPGAN_ckpt,
        device="cuda",
    )
    demo.queue()
    demo.launch(share=args.share, server_name=args.server, server_port=args.port)
