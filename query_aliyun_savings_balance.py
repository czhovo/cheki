import argparse
import base64
import hashlib
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import certifi
import requests


API_ENDPOINT = "https://business.aliyuncs.com/"
API_VERSION = "2017-12-14"
DEFAULT_CREDENTIALS = Path(__file__).with_name(
    ".aliyun_balance_credentials.json"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query the remaining amount of active Alibaba Cloud savings plans."
    )
    parser.add_argument(
        "--credentials",
        type=Path,
        default=DEFAULT_CREDENTIALS,
        help="UTF-8 JSON file containing access_key_id and access_key_secret.",
    )
    parser.add_argument(
        "--instance-id",
        help="Only return the savings plan with this instance ID.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON instead of only the remaining amount.",
    )
    return parser.parse_args()


def percent_encode(value):
    return quote(str(value), safe="~")


def sign_request(parameters, access_key_secret):
    canonical_query = "&".join(
        f"{percent_encode(key)}={percent_encode(parameters[key])}"
        for key in sorted(parameters)
    )
    string_to_sign = f"GET&%2F&{percent_encode(canonical_query)}"
    digest = hmac.new(
        f"{access_key_secret}&".encode("utf-8"),
        string_to_sign.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    return base64.b64encode(digest).decode("ascii")


def query_savings_plans(credentials, instance_id=None):
    parameters = {
        "AccessKeyId": credentials["access_key_id"],
        "Action": "QuerySavingsPlansInstance",
        "Format": "JSON",
        "PageNum": 1,
        "PageSize": 100,
        "SignatureMethod": "HMAC-SHA1",
        "SignatureNonce": str(uuid4()),
        "SignatureVersion": "1.0",
        "Status": "NORMAL",
        "Timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "Version": API_VERSION,
    }
    if instance_id:
        parameters["InstanceId"] = instance_id
    parameters["Signature"] = sign_request(
        parameters, credentials["access_key_secret"]
    )

    response = requests.get(
        API_ENDPOINT,
        params=parameters,
        timeout=30,
        verify=certifi.where(),
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("Success") or payload.get("Code") != "Success":
        raise RuntimeError(
            f"Alibaba Cloud API error: {payload.get('Code')}: "
            f"{payload.get('Message')}"
        )

    items = payload.get("Data", {}).get("Items", [])
    return [
        {
            "instance_id": item.get("InstanceId"),
            "currency": item.get("Currency"),
            "remaining_amount": item.get("RestPoolValue"),
            "status": item.get("Status"),
        }
        for item in items
    ]


def main():
    args = parse_args()
    credentials = json.loads(
        args.credentials.resolve().read_text(encoding="utf-8")
    )
    plans = query_savings_plans(credentials, args.instance_id)
    if not plans:
        raise RuntimeError("No active savings plan instance was returned.")

    if args.json or len(plans) != 1:
        print(json.dumps(plans, ensure_ascii=False, indent=2))
    else:
        print(plans[0]["remaining_amount"])


if __name__ == "__main__":
    main()
