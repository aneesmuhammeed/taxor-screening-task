"""Generate synthetic handwritten-style Indian bill images for testing.

Creates 10 bill images with varied content (vendor names, amounts, dates, 
GST details) and a corresponding ground_truth.json file.

Run this script once to populate data/bills/ and data/ground_truth.json.
These are meant as working test images — replace with real handwritten bills
for meaningful evaluation.
"""

import json
import random
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Output directories
BILLS_DIR = Path(__file__).parent.parent / "data" / "bills"
GT_PATH = Path(__file__).parent.parent / "data" / "ground_truth.json"

# Bill data — each entry will become one image + one ground truth entry
BILLS = [
    {
        "vendor_name": "Sri Lakshmi Stores",
        "invoice_number": "SLS-2024-0451",
        "date": "2024-11-15",
        "date_raw": "15/11/2024",
        "amount": 1250.00,
        "currency": "INR",
        "gst_details": "GSTIN: 29AABCS1429B1ZS, CGST 9%: Rs.56.25, SGST 9%: Rs.56.25",
        "date_ambiguous": False,
    },
    {
        "vendor_name": "Balaji Electronics",
        "invoice_number": "BE-7823",
        "date": "2024-10-03",
        "date_raw": "03/10/2024",
        "amount": 15499.00,
        "currency": "INR",
        "gst_details": "GSTIN: 27AABCB5678K1Z5, Total GST 18%: Rs.2789.82",
        "date_ambiguous": False,
    },
    {
        "vendor_name": "Annapurna Restaurant",
        "invoice_number": None,
        "date": "2024-12-25",
        "date_raw": "25-12-24",
        "amount": 480.00,
        "currency": "INR",
        "gst_details": None,
        "date_ambiguous": False,
    },
    {
        "vendor_name": "Mehta Medical Hall",
        "invoice_number": "MMH/1209",
        "date": "2024-09-18",
        "date_raw": "18/09/2024",
        "amount": 3275.50,
        "currency": "INR",
        "gst_details": "GSTIN: 24AADCM7834N1ZK, GST 12%: Rs.393.06",
        "date_ambiguous": False,
    },
    {
        "vendor_name": "Sharma Cloth Emporium",
        "invoice_number": "SCE-0098",
        "date": "2024-08-07",
        "date_raw": "7/8/2024",
        "amount": 8900.00,
        "currency": "INR",
        "gst_details": "GSTIN: 09AALCS4321P1ZQ, CGST 2.5%: Rs.111.25, SGST 2.5%: Rs.111.25",
        "date_ambiguous": False,
    },
    {
        "vendor_name": "Gupta Hardware",
        "invoice_number": "GH-445",
        "date": "2024-07-22",
        "date_raw": "22-07-24",
        "amount": 2150.00,
        "currency": "INR",
        "gst_details": None,
        "date_ambiguous": False,
    },
    {
        "vendor_name": "Ravi Xerox & Stationery",
        "invoice_number": None,
        "date": "2024-11-02",
        "date_raw": "2/11/24",
        "amount": 185.00,
        "currency": "INR",
        "gst_details": None,
        "date_ambiguous": False,
    },
    {
        "vendor_name": "New Bombay Sweets",
        "invoice_number": "NBS-3344",
        "date": "2024-12-31",
        "date_raw": "31/12/2024",
        "amount": 750.00,
        "currency": "INR",
        "gst_details": "GSTIN: 19AABCN2233L1ZE, GST 5%: Rs.37.50",
        "date_ambiguous": False,
    },
    {
        "vendor_name": "Sai Auto Parts",
        "invoice_number": "SA/2024/789",
        "date": "2024-06-14",
        "date_raw": "14-6-2024",
        "amount": 4320.00,
        "currency": "INR",
        "gst_details": "GSTIN: 36AADCS9876Q1ZJ, Total Tax 28%: Rs.1209.60",
        "date_ambiguous": False,
    },
    {
        "vendor_name": "Patel Kirana Store",
        "invoice_number": "PKS-112",
        "date": None,
        "date_raw": "1?/03/24",
        "amount": 567.00,
        "currency": "INR",
        "gst_details": None,
        "date_ambiguous": True,
    },
]


def draw_bill_image(bill: dict, index: int) -> Path:
    """Create a synthetic bill image that looks like a handwritten receipt.

    Uses basic PIL drawing since we can't rely on handwriting fonts being
    installed. The images simulate lined paper with text at slightly varied
    positions to mimic handwriting irregularity.
    """
    width, height = 600, 800
    # Slightly off-white background to look like receipt paper
    bg_color = random.choice([(255, 252, 240), (245, 245, 235), (250, 248, 230)])
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Use default font (no external dependency)
    try:
        font_large = ImageFont.truetype("arial.ttf", 24)
        font_medium = ImageFont.truetype("arial.ttf", 18)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except (IOError, OSError):
        font_large = ImageFont.load_default()
        font_medium = font_large
        font_small = font_large

    # Ink color — slightly varied to simulate pen
    ink = random.choice([(10, 10, 80), (0, 0, 0), (30, 30, 120)])

    y = 30

    # Draw a border
    draw.rectangle([15, 15, width - 15, height - 15], outline=ink, width=2)

    # Vendor name (centered, larger)
    vendor = bill["vendor_name"]
    bbox = draw.textbbox((0, 0), vendor, font=font_large)
    text_width = bbox[2] - bbox[0]
    x_center = (width - text_width) // 2 + random.randint(-5, 5)
    draw.text((x_center, y), vendor, fill=ink, font=font_large)
    y += 40

    # Underline
    draw.line([(40, y), (width - 40, y)], fill=ink, width=1)
    y += 20

    # Invoice number
    if bill["invoice_number"]:
        inv_text = f"Bill No: {bill['invoice_number']}"
        draw.text((30 + random.randint(0, 5), y), inv_text, fill=ink, font=font_medium)
        y += 30

    # Date
    if bill["date_raw"]:
        date_text = f"Date: {bill['date_raw']}"
        draw.text((30 + random.randint(0, 3), y), date_text, fill=ink, font=font_medium)
        y += 30

    y += 10
    draw.line([(30, y), (width - 30, y)], fill=ink, width=1)
    y += 15

    # Some item lines to make it look like a real bill
    items = [
        ("Item 1", random.randint(50, 500)),
        ("Item 2", random.randint(30, 300)),
        ("Item 3", random.randint(20, 200)),
    ]
    remaining = bill["amount"]
    for i, (item_name, _) in enumerate(items[:-1]):
        item_amt = round(remaining * random.uniform(0.2, 0.5), 2)
        remaining -= item_amt
        line = f"  {item_name}{'.' * 30} Rs.{item_amt:.2f}"
        draw.text(
            (30 + random.randint(0, 3), y),
            line,
            fill=ink,
            font=font_small,
        )
        y += 25

    # Last item gets the remainder
    line = f"  {items[-1][0]}{'.' * 30} Rs.{remaining:.2f}"
    draw.text((30 + random.randint(0, 3), y), line, fill=ink, font=font_small)
    y += 35

    # Total line
    draw.line([(30, y), (width - 30, y)], fill=ink, width=1)
    y += 10
    total_text = f"TOTAL: Rs. {bill['amount']:.2f}"
    bbox = draw.textbbox((0, 0), total_text, font=font_large)
    text_width = bbox[2] - bbox[0]
    draw.text(
        ((width - text_width) // 2 + random.randint(-3, 3), y),
        total_text,
        fill=ink,
        font=font_large,
    )
    y += 45

    # GST details
    if bill["gst_details"]:
        draw.line([(30, y), (width - 30, y)], fill=ink, width=1)
        y += 10
        # Wrap long GST text
        gst_lines = textwrap.wrap(bill["gst_details"], width=50)
        for gst_line in gst_lines:
            draw.text(
                (30 + random.randint(0, 3), y),
                gst_line,
                fill=ink,
                font=font_small,
            )
            y += 20

    # Add some noise dots to simulate paper texture
    for _ in range(random.randint(30, 80)):
        nx = random.randint(20, width - 20)
        ny = random.randint(20, height - 20)
        draw.point((nx, ny), fill=(180, 170, 150))

    # Save
    filename = f"bill_{index + 1:02d}.png"
    filepath = BILLS_DIR / filename
    img.save(filepath)
    return filepath


def generate_dataset():
    """Generate all bill images and ground truth file."""
    BILLS_DIR.mkdir(parents=True, exist_ok=True)

    ground_truth = []

    for i, bill in enumerate(BILLS):
        filepath = draw_bill_image(bill, i)
        print(f"  Created: {filepath.name}")

        # Build ground truth entry
        gt_entry = {
            "image_file": filepath.name,
            "vendor_name": bill["vendor_name"],
            "invoice_number": bill["invoice_number"],
            "date": bill["date"],
            "date_raw": bill["date_raw"],
            "date_ambiguous": bill["date_ambiguous"],
            "amount": bill["amount"],
            "currency": bill["currency"],
            "gst_details": bill["gst_details"],
        }
        ground_truth.append(gt_entry)

    # Write ground truth
    GT_PATH.write_text(json.dumps(ground_truth, indent=2, ensure_ascii=False))
    print(f"\n  Ground truth written to: {GT_PATH}")
    print(f"  Total bills: {len(ground_truth)}")


if __name__ == "__main__":
    print("Generating synthetic bill dataset...")
    generate_dataset()
    print("Done!")
