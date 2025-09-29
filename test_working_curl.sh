#!/bin/bash

echo "🧪 Testing the working curl command from user example..."
echo "=================================================="

curl --location 'https://hp-buyer-backend-preprod.himira.co.in/clientApis/v2/select' \
--header 'Accept: application/json, text/plain, */*' \
--header 'Content-Type: application/json' \
--header 'wil-api-key: aPzSpx0rksO96PhGGNKRgfAay0vUbZ' \
--data '[
    {
        "context": {
            "domain": "ONDC:RET10",
            "city": "140301"
        },
        "message": {
            "cart": {
                "items": [
                    {
                        "id": "hp-seller-preprod.himira.co.in_ONDC:RET10_d871c2ae-bf3f-4d3c-963f-f85f94848e8c_8b3ac3ad-a0e0-46cc-ac30-11357615c877",
                        "local_id": "8b3ac3ad-a0e0-46cc-ac30-11357615c877",
                        "customisationState": {},
                        "quantity": {
                            "count": 1
                        },
                        "provider": {
                            "id": "hp-seller-preprod.himira.co.in_ONDC:RET10_d871c2ae-bf3f-4d3c-963f-f85f94848e8c",
                            "local_id": "d871c2ae-bf3f-4d3c-963f-f85f94848e8c",
                            "locations": [
                                {
                                    "id": "hp-seller-preprod.himira.co.in_ONDC:RET10_d871c2ae-bf3f-4d3c-963f-f85f94848e8c_d871c2ae-bf3f-4d3c-963f-f85f94848e8c",
                                    "local_id": "d871c2ae-bf3f-4d3c-963f-f85f94848e8c"
                                }
                            ]
                        },
                        "customisations": null,
                        "hasCustomisations": false
                    }
                ]
            },
            "fulfillments": [
                {
                    "end": {
                        "location": {
                            "gps": "30.745765,76.653633",
                            "address": {
                                "area_code": "140301"
                            }
                        }
                    }
                }
            ]
        },
        "userId": "fbw2k45A8cXMGf8MoszQ8pymEzT2",
        "deviceId": "75023e93265cca67a0e8c6b67155716d"
    }
]' | jq . || echo "❌ Curl command failed or response not valid JSON"