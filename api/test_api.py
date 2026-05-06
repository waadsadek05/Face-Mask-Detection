"""
test_api.py — Face Mask Detection API Tests
==========================================
Run AFTER the API is running:
    uvicorn app:app --reload --port 8000
Then:
    python test_api.py
"""

import requests
import sys
import json
from pathlib import Path
from PIL import Image
import io

API_URL = "http://localhost:8000"

# ── Helpers ──────────────────────────────────────────────────────────────────

def create_dummy_image(color: tuple = (200, 200, 200), size=(224, 224)) -> bytes:
    """Create a small in-memory PNG for tests that don't need a real photo."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def post_image(image_bytes: bytes, filename: str = "test.png",
               content_type: str = "image/png") -> requests.Response:
    files = {"file": (filename, image_bytes, content_type)}
    return requests.post(f"{API_URL}/predict", files=files, timeout=30)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_health_check():
    """GET / should respond with 200."""
    print("\n[1] Health check …")
    try:
        r = requests.get(f"{API_URL}/", timeout=5)
        print(f"    GET /  →  status {r.status_code}  |  {r.json()}")
        assert r.status_code == 200
        print("    ✅  Passed")
    except requests.exceptions.ConnectionError:
        print("    ❌  Cannot connect — is the API running?")
        sys.exit(1)


def test_predict_with_dummy_image():
    """POST /predict should return valid JSON for any RGB image."""
    print("\n[2] Predict with dummy (grey) image …")
    r = post_image(create_dummy_image())

    assert r.status_code == 200, f"Expected 200, got {r.status_code}\n{r.text}"
    body = r.json()
    print(f"    Response: {json.dumps(body, indent=6)}")

    assert "status"     in body, "Missing field: status"
    assert "action"     in body, "Missing field: action"
    assert "confidence" in body, "Missing field: confidence"
    assert body["status"] in ("mask_on", "mask_off"),          f"Unknown status: {body['status']}"
    assert body["action"] in ("Allow entry", "Deny entry"),    f"Unknown action: {body['action']}"
    assert 0.0 <= body["confidence"] <= 1.0,                   "Confidence out of range"
    print("    ✅  Passed")


def test_predict_with_real_images():
    """
    If real test images exist under data/test/with_mask or data/test/without_mask,
    run predictions and print per-class accuracy.
    """
    print("\n[3] Predict with real test images …")

    label_map = {
        "with_mask":    "mask_on",
        "without_mask": "mask_off",
    }

    found_any = False
    all_results = []

    for label, expected_status in label_map.items():
        folder = Path(f"data/test/{label}")
        if not folder.exists():
            continue
        images = (list(folder.glob("*.jpg")) + list(folder.glob("*.png")))[:5]
        if not images:
            continue

        found_any = True
        correct = 0
        for img_path in images:
            with open(img_path, "rb") as f:
                r = post_image(f.read(), img_path.name)
            assert r.status_code == 200, f"Got {r.status_code} for {img_path.name}"
            body = r.json()
            ok = body["status"] == expected_status
            correct += int(ok)
            all_results.append((img_path.name, expected_status, body["status"],
                                 body["confidence"], ok))

        acc = correct / len(images) * 100
        print(f"    {label}: {correct}/{len(images)} correct  ({acc:.0f}%)")

    if not found_any:
        print("    ⚠️  No real test images found — skipping (not a failure).")
        return

    print()
    for name, expected, got, conf, ok in all_results:
        icon = "✅" if ok else "❌"
        print(f"    {icon}  {name:<30} expected={expected:<8}  got={got:<8}  conf={conf:.2f}")


def test_invalid_file_rejected():
    """Sending a plain-text file should be rejected (not return 200)."""
    print("\n[4] Invalid file (plain text) …")
    fake = b"this is definitely not an image"
    r = post_image(fake, filename="not_image.txt", content_type="text/plain")
    print(f"    Status code: {r.status_code}")
    assert r.status_code != 200, \
        "API accepted a non-image — it should have rejected it with 4xx/5xx"
    print("    ✅  Passed (non-image correctly rejected)")


def test_response_fields_types():
    """Verify exact Python types of the response fields."""
    print("\n[5] Response field types …")
    r = post_image(create_dummy_image())
    body = r.json()

    assert isinstance(body["status"],     str),   "status must be str"
    assert isinstance(body["action"],     str),   "action must be str"
    assert isinstance(body["confidence"], float), "confidence must be float"
    print(f"    status={type(body['status']).__name__}  "
          f"action={type(body['action']).__name__}  "
          f"confidence={type(body['confidence']).__name__}")
    print("    ✅  Passed")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Face Mask Detection — API Test Suite")
    print("=" * 55)

    test_health_check()
    test_predict_with_dummy_image()
    test_predict_with_real_images()
    test_invalid_file_rejected()
    test_response_fields_types()

    print("\n" + "=" * 55)
    print("  All tests passed! ✅")
    print("=" * 55)