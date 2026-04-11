import gradio as gr
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
import io
from datetime import datetime
import os

print("="*60)
print("🌾 KHARPATNAASHAK - WEED DETECTION SYSTEM")
print("="*60)

# Load model from current directory
model_path = "best.pt"
if not os.path.exists(model_path):
    # Fallback to any .pt file in the directory
    import glob
    pt_files = glob.glob("*.pt")
    if pt_files:
        model_path = pt_files[0]
        print(f"📁 Found model: {model_path}")

print(f"🤖 Loading model from: {model_path}")
model = YOLO(model_path)
print("✅ Model loaded!")

# Weed names
WEED_NAMES_EN = {
    'Kena': 'Kena', 'Lavhala': 'Lavhala', 'Lambs Quarter Plant': 'Lambs Quarter Plant',
    'Little Mallow': 'Little Mallow', 'Moti_dudhi': 'Moti_dudhi',
    'Obscure morning glory': 'Obscure morning glory', 'Asian pigeon wings': 'Asian pigeon wings',
    'Bilayat': 'Bilayat', 'Choti_dudhi': 'Choti_dudhi', 'Digitaria SP': 'Digitaria SP',
    'Gajar_gavat': 'Gajar_gavat', 'Gracefulsandmat': 'Gracefulsandmat',
    'Sicklepod': 'Sicklepod', 'Harali': 'Harali', 'Dwarf Cassia': 'Dwarf Cassia', 'Punarnava': 'Punarnava'
}

WEED_NAMES_HI = {
    'Kena': 'केना', 'Lavhala': 'लव्हाळा', 'Lambs Quarter Plant': 'बथुआ',
    'Little Mallow': 'छोटी खरपतवार', 'Moti_dudhi': 'मोती दूधी',
    'Obscure morning glory': 'अस्पष्ट मॉर्निंग ग्लोरी', 'Asian pigeon wings': 'एशियाई कबूतर पंख',
    'Bilayat': 'बिलायत', 'Choti_dudhi': 'छोटी दूधी', 'Digitaria SP': 'डिजिटेरिया',
    'Gajar_gavat': 'गाजर गवत', 'Gracefulsandmat': 'ग्रेसफुलसैंडमैट',
    'Sicklepod': 'सिकलपॉड', 'Harali': 'हराली', 'Dwarf Cassia': 'बौनी कैसिया', 'Punarnava': 'पुनर्नवा'
}

TEXTS = {
    'en': {'title': '🌾 KharpatNaashak — Weed Detection System for Madhya Pradesh',
           'detect_btn': '🔍 Detect Weeds', 'confidence': '🎯 Confidence Threshold',
           'heatmap': '🗺️ Show Weed Density Map', 'field_area': '📏 Field Area (acres)',
           'savings': '💰 Cost Savings', 'weed_breakdown': '🌿 Weed Breakdown',
           'no_weeds': '✅ No weeds detected', 'traditional': 'Traditional Spraying',
           'precision': 'Precision Spraying', 'your_savings': 'Your Savings',
           'total_weeds': 'Total Weeds Detected', 'avg_confidence': 'Average Confidence'},
    'hi': {'title': '🌾 खरपतवार नाशक — मध्य प्रदेश के लिए खरपतवार पहचान प्रणाली',
           'detect_btn': '🔍 खरपतवार पहचानें', 'confidence': '🎯 विश्वास सीमा',
           'heatmap': '🗺️ खरपतवार घनत्व मानचित्र दिखाएं', 'field_area': '📏 खेत का क्षेत्रफल (एकड़)',
           'savings': '💰 लागत बचत', 'weed_breakdown': '🌿 खरपतवार विवरण',
           'no_weeds': '✅ कोई खरपतवार नहीं मिला', 'traditional': 'पारंपरिक छिड़काव',
           'precision': 'सटीक छिड़काव', 'your_savings': 'आपकी बचत',
           'total_weeds': 'कुल खरपतवार', 'avg_confidence': 'औसत विश्वास'}
}

def get_text(key, lang):
    return TEXTS[lang].get(key, TEXTS['en'][key])

def calculate_savings(weed_count, field_area=1):
    traditional = field_area * 800
    precision = weed_count * 5
    savings = traditional - precision
    percent = (savings / traditional) * 100 if traditional > 0 else 0
    return traditional, precision, savings, percent

def create_heatmap(image, results, grid_size=10):
    h, w = image.shape[:2]
    cell_h, cell_w = h // grid_size, w // grid_size
    density = np.zeros((grid_size, grid_size))
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        gx, gy = min(cx // cell_w, grid_size-1), min(cy // cell_h, grid_size-1)
        density[gy, gx] += 1
    if density.max() > 0:
        density = density / density.max()
    heatmap = plt.cm.YlOrRd(density)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.resize(heatmap, (w, h))
    return cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

def generate_pdf_report(detections_list, total_weeds, savings_data, farmer_name="", field_name=""):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(50, height - 50, "KharpatNaashak")
    pdf.setFont("Helvetica", 12)
    pdf.drawString(50, height - 70, "Weed Detection Report")
    
    pdf.setFont("Helvetica", 10)
    pdf.drawString(50, height - 100, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if farmer_name:
        pdf.drawString(50, height - 115, f"Farmer: {farmer_name}")
    if field_name:
        pdf.drawString(50, height - 130, f"Field: {field_name}")
    
    y_pos = height - 170
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y_pos, "Detection Summary")
    y_pos -= 25
    
    pdf.setFont("Helvetica", 12)
    pdf.drawString(50, y_pos, f"Total Weeds Detected: {total_weeds}")
    y_pos -= 20
    pdf.drawString(50, y_pos, f"Traditional Spraying Cost: ₹{savings_data[0]:,.0f}")
    y_pos -= 20
    pdf.drawString(50, y_pos, f"Precision Spraying Cost: ₹{savings_data[1]:,.0f}")
    y_pos -= 20
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y_pos, f"Estimated Savings: ₹{savings_data[2]:,.0f} ({savings_data[3]:.0f}% reduction)")
    y_pos -= 30
    
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y_pos, "Weed Breakdown")
    y_pos -= 25
    
    pdf.setFont("Helvetica", 10)
    for name, count in detections_list[:25]:
        pdf.drawString(50, y_pos, f"{name}: {count}")
        y_pos -= 15
        if y_pos < 50:
            pdf.showPage()
            y_pos = height - 50
    
    pdf.save()
    buffer.seek(0)
    return buffer

def process_image(image, confidence=0.25, show_heatmap=False, field_area=1, lang='en'):
    results = model(image, conf=confidence)
    annotated = results[0].plot()
    detections, weed_counts, confs = [], {}, []
    
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cid, conf = int(box.cls[0]), float(box.conf[0])
            name = model.names[cid]
            display = WEED_NAMES_HI[name] if lang == 'hi' else WEED_NAMES_EN[name]
            detections.append(f"{display}: {conf:.1%}")
            weed_counts[name] = weed_counts.get(name, 0) + 1
            confs.append(conf)
    
    total = len(detections)
    avg = sum(confs)/len(confs) if confs else 0
    trad, prec, save, pct = calculate_savings(total, field_area)
    
    summary = f"""### 💰 Cost Savings (per {field_area} acre)
| Method | Cost (₹) |
|--------|---------|
| Traditional Spraying | ₹{trad:,.0f} |
| Precision Spraying | ₹{prec:,.0f} |
| **Your Savings** | **₹{save:,.0f}** ({pct:.0f}% reduction) |

### 🌿 Weed Breakdown
"""
    for name, count in sorted(weed_counts.items(), key=lambda x: x[1], reverse=True):
        display = WEED_NAMES_HI[name] if lang == 'hi' else WEED_NAMES_EN[name]
        summary += f"- **{display}**: {count}\n"
    if total == 0:
        summary += f"\n{get_text('no_weeds', lang)}"
    
    heatmap_img = create_heatmap(np.array(image), results) if show_heatmap and total > 0 else None
    stats = f"📊 {get_text('total_weeds', lang)}: {total} | {get_text('avg_confidence', lang)}: {avg:.1%}"
    detections_text = "\n".join(detections) if detections else get_text('no_weeds', lang)
    
    return annotated, heatmap_img, summary, detections_text, stats, weed_counts

def process_video(video_file, confidence=0.25, lang='en'):
    cap = cv2.VideoCapture(video_file.name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    fc, total = 0, 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        fc += 1
        results = model(frame, conf=confidence)
        out.write(results[0].plot())
        total += len(results[0].boxes)
    
    cap.release()
    out.release()
    return out_path, f"✅ Processed {fc} frames | Total weeds: {total}"

last_results = {"detections": [], "total": 0, "savings": (0,0,0,0)}

def process_multiple_with_details(files, conf, heatmap, field_area=1):
    gallery_images = []
    heatmap_images = []
    all_detections = {}
    total_weeds_all = 0
    
    for f in files:
        img = Image.open(f.name)
        annotated, heatmap_img, summary, detections, stats, weed_counts = process_image(img, conf, heatmap, field_area, 'en')
        gallery_images.append((annotated, f.name))
        if heatmap_img is not None:
            heatmap_images.append((heatmap_img, f"Heatmap - {f.name}"))
        
        for name, count in weed_counts.items():
            all_detections[name] = all_detections.get(name, 0) + count
        det_count = len(detections.split('\n')) if detections != 'No weeds detected' else 0
        total_weeds_all += det_count
    
    trad, prec, save, pct = calculate_savings(total_weeds_all, field_area)
    
    summary_text = f"""### 📊 TOTAL SUMMARY
**Total Images:** {len(files)}
**Total Weeds Detected:** {total_weeds_all}

### 💰 COST SAVINGS (for {field_area} acre)
| Method | Cost (₹) |
|--------|---------|
| Traditional Spraying | ₹{trad:,.0f} |
| Precision Spraying | ₹{prec:,.0f} |
| **Your Savings** | **₹{save:,.0f}** ({pct:.0f}% reduction) |

### 🌿 WEED BREAKDOWN (All Images Combined)
"""
    for name, count in sorted(all_detections.items(), key=lambda x: x[1], reverse=True):
        summary_text += f"- **{WEED_NAMES_EN.get(name, name)}**: {count}\n"
    if total_weeds_all == 0:
        summary_text += "No weeds detected in any image"
    
    last_results["detections"] = [(WEED_NAMES_EN.get(name, name), count) for name, count in sorted(all_detections.items(), key=lambda x: x[1], reverse=True)]
    last_results["total"] = total_weeds_all
    last_results["savings"] = (trad, prec, save, pct)
    
    all_gallery = gallery_images + heatmap_images
    return all_gallery, summary_text

print("\n🎨 Launching KharpatNaashak...")

with gr.Blocks(title="KharpatNaashak - Weed Detection", theme=gr.themes.Soft()) as demo:
    with gr.Row():
        lang_selector = gr.Radio(choices=["English", "हिंदी"], label="🌐 Language / भाषा", value="English")

    gr.HTML("""<div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #2E7D32, #1B5E20); border-radius: 10px;">
        <h1 style="color: white;">🌾 खरपतवार नाशक</h1>
        <p style="color: #C8E6C9;">KharpatNaashak — Weed Detection System for Madhya Pradesh</p>
        <p style="color: #FFD700; font-size: 0.9rem;">Model: 100 Epochs | mAP@0.5: 60.7% | Precision: 67.1% | Recall: 52.2%</p>
    </div>""")

    with gr.Tabs():
        with gr.TabItem("📸 Single Image"):
            with gr.Row():
                with gr.Column():
                    img_in = gr.Image(type="pil", label="Upload Field Image", height=300)
                    conf_slider = gr.Slider(0.1, 0.9, 0.25, step=0.05, label="Confidence Threshold")
                    with gr.Row():
                        heatmap_chk = gr.Checkbox(False, label="Show Weed Density Map")
                        field_area_single = gr.Number(value=1, label="Field Area (acres)", step=0.5)
                    detect_btn = gr.Button("Detect Weeds", variant="primary", size="lg")
                with gr.Column():
                    img_out = gr.Image(label="Detection Results", height=300)
                    heatmap_out = gr.Image(label="Weed Density Map", height=300)
            with gr.Row():
                with gr.Column(): stats_out = gr.Markdown("")
                with gr.Column(): summary_out = gr.Markdown("")
            detections_out = gr.Textbox(label="Detected Weeds", lines=6)

        with gr.TabItem("🎬 Video"):
            with gr.Row():
                with gr.Column():
                    video_in = gr.File(label="Upload Video (MP4)", file_types=[".mp4", ".avi", ".mov"])
                    video_conf = gr.Slider(0.1, 0.9, 0.25, step=0.05, label="Confidence Threshold")
                    process_vid_btn = gr.Button("Process Video", variant="primary")
                with gr.Column():
                    video_out = gr.Video(label="Processed Video")
                    video_summary = gr.Textbox(label="Summary", lines=3)

        with gr.TabItem("📚 Multiple Images"):
            with gr.Row():
                with gr.Column():
                    files_in = gr.File(file_count="multiple", label="Upload Multiple Images")
                    multi_conf = gr.Slider(0.1, 0.9, 0.25, step=0.05, label="Confidence Threshold")
                    multi_heatmap = gr.Checkbox(False, label="Show Heatmaps")
                    multi_area = gr.Number(value=1, label="Field Area (acres)", step=0.5)
                    process_all_btn = gr.Button("Process All", variant="primary")
                    with gr.Row():
                        farmer_name = gr.Textbox(label="👨‍🌾 Farmer Name", placeholder="Enter farmer name")
                        field_name = gr.Textbox(label="🌾 Field Name", placeholder="Enter field name")
                    report_btn = gr.Button("📄 Generate PDF Report", variant="secondary")
                    pdf_output = gr.File(label="Download Report")
                with gr.Column():
                    gallery_out = gr.Gallery(label="Results (with Heatmaps)", columns=2, height=500)
                    multi_summary = gr.Markdown("")

        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
            ### 🌾 KharpatNaashak - Weed Detection System
            
            **Model:** YOLOv11 trained on MH-Weed16 dataset
            **Training:** 100 epochs
            **Performance:** mAP@0.5: 60.7% | Precision: 67.1% | Recall: 52.2%
            
            ### 🌿 16 Weed Species Detected
            Kena, Lavhala, Lambs Quarter Plant, Little Mallow, Moti_dudhi,
            Obscure morning glory, Asian pigeon wings, Bilayat, Choti_dudhi,
            Digitaria SP, Gajar_gavat, Gracefulsandmat, Sicklepod, Harali,
            Dwarf Cassia, Punarnava
            
            ### 🚀 Features
            - ✅ Hindi/English toggle
            - ✅ Video upload & processing
            - ✅ Weed density heatmaps
            - ✅ Cost savings calculator
            - ✅ Multiple image upload with heatmaps
            - ✅ PDF report generation with farmer details
            """)

    def detect_wrapper(img, conf, heatmap, area, lang):
        return process_image(img, conf, heatmap, area, 'hi' if lang == 'हिंदी' else 'en')
    
    detect_btn.click(detect_wrapper, [img_in, conf_slider, heatmap_chk, field_area_single, lang_selector], [img_out, heatmap_out, summary_out, detections_out, stats_out])
    
    def video_wrapper(vid, conf, lang):
        return process_video(vid, conf, 'hi' if lang == 'हिंदी' else 'en')
    
    process_vid_btn.click(video_wrapper, [video_in, video_conf, lang_selector], [video_out, video_summary])
    
    def process_and_display(files, conf, heatmap, area):
        return process_multiple_with_details(files, conf, heatmap, area)
    
    process_all_btn.click(process_and_display, [files_in, multi_conf, multi_heatmap, multi_area], [gallery_out, multi_summary])
    
    def generate_report(farmer, field):
        if last_results["detections"]:
            return generate_pdf_report(last_results["detections"], last_results["total"], last_results["savings"], farmer, field)
        else:
            buffer = io.BytesIO()
            pdf = canvas.Canvas(buffer, pagesize=letter)
            pdf.drawString(50, 750, "No weed detection data available. Please process images first.")
            pdf.save()
            buffer.seek(0)
            return buffer
    
    report_btn.click(generate_report, [farmer_name, field_name], [pdf_output])

demo.launch()
