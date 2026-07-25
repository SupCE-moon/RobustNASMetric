from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path

import requests


ROBUSTBENCH_CIFAR10_LINF_IDS = {
    "Standard": "1t98aEuzeTL8P7Kpd5DIrCoCL21BNZUhC",
    "Rice2020Overfitting": "1vC_Twazji7lBjeMQvAD9uEQxi9Nx2oG-",
    "Wong2020Fast": "1Re--_lf3jCEw9bnQqGkjw3J7v2tSZKrv",
    "Engstrom2019Robustness": "1etqmQsksNIWBvBQ4r8ZFk_3FJlLWr8Rr",
    "Hendrycks2019Using": "1-DcJsYw2dNEOyF9epks2QS7r9nqBDEsw",
    "Zhang2019Theoretically": "1hPz9QQwyM7QSuWu-ANG_uXR-29xtL8t_",
    "Carmon2019Unlabeled": "15tUx-gkZMYx7BfEOw1GY5OKC-jECIsPQ",
    "Sehwag2020Hydra": "1pi8GHwAVkxVH41hEnf0IAJb_7y-Q8a2Y",
    "Sehwag2021Proxy_R18": "1-ZgoSlD_AMhtXdnUElilxVXnzK2DcHuu",
    "Addepalli2022Efficient_RN18": "1m5vhdzIUUKhDbsZdOG9z76Eyp6f4xe_f",
    "Huang2020Self": "1nInDeIyZe2G-mJFxQJ3UoclQNomWjMgm",
    "Pang2020Boosting": "1iNWOj3MP7kGe8yTAS4XnDaDXDLt0mwqw",
}


class DriveFormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_form = False
        self.action: str | None = None
        self.params: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self.in_form = True
            self.action = attrs.get("action")
        elif self.in_form and tag == "input" and attrs.get("name"):
            self.params[attrs["name"]] = attrs.get("value", "")

    def handle_endtag(self, tag: str) -> None:
        if tag == "form":
            self.in_form = False


def _write_stream(response: requests.Response, output: Path) -> None:
    total = 0
    with output.open("wb") as f:
        for chunk in response.iter_content(1024 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            total += len(chunk)
            if total % (50 * 1024 * 1024) < 1024 * 1024:
                print(f"downloaded={total}")


def download_gdrive(file_id: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    response = session.get(
        "https://drive.google.com/uc",
        params={"export": "download", "id": file_id},
        stream=True,
        timeout=60,
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type:
        _write_stream(response, output)
        return

    parser = DriveFormParser()
    parser.feed(response.text)
    if not parser.action:
        raise RuntimeError("Google Drive confirmation form was not found.")

    response = session.get(parser.action, params=parser.params, stream=True, timeout=60)
    response.raise_for_status()
    _write_stream(response, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="Model names or raw Google Drive IDs.")
    parser.add_argument("--output-dir", default="models/cifar10/Linf")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for item in args.models:
        file_id = ROBUSTBENCH_CIFAR10_LINF_IDS.get(item, item)
        name = item if item in ROBUSTBENCH_CIFAR10_LINF_IDS else file_id
        output = output_dir / f"{name}.pt"
        if output.exists() and output.stat().st_size > 10 * 1024 * 1024:
            print(f"skip={output}")
            continue
        print(f"download={name}")
        download_gdrive(file_id, output)
        print(f"saved={output} bytes={output.stat().st_size}")


if __name__ == "__main__":
    main()
