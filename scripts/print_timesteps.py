from PIL import Image, ImageDraw
import os

results_dir = r"C:\Users\Admin\Desktop\csci484\surrogate_modeling\MORPH\experiments\results\test\fns-kf"
prefix = "ft_st_MORPH-Ti_FM_ar1_chAll_samp0_tot-trajs100_tot-eps0_rank-lora16_ftlevel1_lr0.0001_wd0.0_t"

images = [Image.open(os.path.join(results_dir, f"{prefix}{t}.png")) for t in range(10)]

img_w, img_h = images[0].size
col_w = img_w // 5
row_h = img_h // 2  # Ch0 is top half, Ch1 is bottom half

target_col = 1
pred_col = 2

for ch, ch_name, row_y in [("Ch0", "ch0", 0), ("Ch1", "ch1", row_h)]:
    label_h = 40
    final_w = col_w * 10
    final_h = row_h * 2 + label_h * 3

    final = Image.new("RGB", (final_w, final_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(final)

    for t in range(10):
        x = t * col_w
        draw.text((x + col_w//2 - 15, 5), f"t={t}", fill=(0, 0, 0))

        target_crop = images[t].crop((target_col * col_w, row_y, (target_col + 1) * col_w, row_y + row_h))
        pred_crop   = images[t].crop((pred_col   * col_w, row_y, (pred_col   + 1) * col_w, row_y + row_h))

        final.paste(target_crop, (x, label_h))
        final.paste(pred_crop,   (x, label_h + row_h + label_h))

    draw.text((5, label_h + row_h//2),            "TARGET",     fill=(0, 0, 0))
    draw.text((5, label_h + row_h + label_h + row_h//2), "PREDICTION", fill=(0, 0, 0))

    out_path = rf"C:\Users\Admin\Desktop\comparison_{ch_name}_t0_t9.png"
    final.save(out_path)
    print(f"Saved: {out_path}")