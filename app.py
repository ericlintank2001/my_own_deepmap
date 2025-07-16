import os
import uuid
import zipfile
import shutil
import threading
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, send_file, jsonify
import gradio as gr
import socket

# ==== MiDaS 初始化 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval().to(device)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

def generate_depth(image: Image.Image) -> Image.Image:
    img_np = np.array(image.convert("RGB"))
    transformed = midas_transforms(img_np)
    input_tensor = (
        transformed["image"].to(device)
        if isinstance(transformed, dict)
        else transformed.to(device)
    )
    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth = prediction.cpu().numpy()
    depth_min, depth_max = np.min(depth), np.max(depth)
    depth_vis = (255 * (depth - depth_min) / (depth_max - depth_min)).astype("uint8")
    return Image.fromarray(depth_vis)

# ==== Gradio 單張圖片處理 ====
def predict_depth(image):
    depth_img = generate_depth(image)
    filename = f"/tmp/depth_{uuid.uuid4().hex}.png"
    depth_img.save(filename)
    return depth_img, filename

# ==== Flask 批次處理 ====
app = Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "online"})

@app.route("/upload", methods=["POST"])
def upload_and_process():
    if "file" not in request.files:
        return "未找到檔案", 400
    file = request.files["file"]
    if not file.filename.endswith(".zip"):
        return "請上傳 ZIP 壓縮檔", 400

    session_id = uuid.uuid4().hex
    input_dir = f"./temp/input_{session_id}"
    output_dir = f"./temp/output_{session_id}"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    zip_path = f"./temp/upload_{session_id}.zip"
    file.save(zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(input_dir)

    for fname in os.listdir(input_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(input_dir, fname)
            output_path = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_depth.png")
            img = Image.open(input_path)
            depth = generate_depth(img)
            depth.save(output_path)

    result_zip = f"./temp/depth_result_{session_id}.zip"
    with zipfile.ZipFile(result_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for f in os.listdir(output_dir):
            zipf.write(os.path.join(output_dir, f), arcname=f)

    shutil.rmtree(input_dir)
    shutil.rmtree(output_dir)
    os.remove(zip_path)

    return send_file(result_zip, as_attachment=True)

# ==== 本地 IP 偵測 ====
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "localhost"

# ==== 啟動 Flask 與 Gradio ====
def start_flask():
    app.run(host="0.0.0.0", port=7861)

def start_gradio():
    local_ip = get_local_ip()
    api_url = f"http://{local_ip}:7861/upload"

    with gr.Blocks(title="2D單圖→深度圖 MiDaS Space") as demo:
        gr.Markdown("### 📷 上傳圖片 → 預覽深度圖 + 下載 PNG")
        input_img = gr.Image(type="pil", label="上傳圖片")
        output_img = gr.Image(type="pil", label="深度圖預覽")
        output_file = gr.File(label="下載深度圖（PNG）")

        input_img.change(fn=predict_depth, inputs=input_img, outputs=[output_img, output_file])

        gr.Markdown("---")

        gr.Markdown(
            f"📡 **本地端程式請使用以下 API 傳送 ZIP 資料夾：**  \n"
            f"`POST {api_url}`  \n"
            "📁 內容為 .zip 檔，伺服器將回傳處理後的深度圖 ZIP。"
        )

    demo.launch(server_port=7860, show_api=True)

# ==== 主程式 ====
if __name__ == "__main__":
    threading.Thread(target=start_flask).start()
    threading.Thread(target=start_gradio).start()
