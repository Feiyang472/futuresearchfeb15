import json

# For parsing WARC records:
import urllib
from dataclasses import dataclass
from typing import ClassVar, NamedTuple
from urllib.error import URLError
from urllib.parse import quote, quote_plus, urlencode, urljoin, urlunparse
from urllib.request import Request, urlopen

import requests
from warcio.archiveiterator import ArchiveIterator
from warcio.recordloader import ArcWarcRecord

from src.constants import all_available_indexes


@dataclass
class CommonCrawlContent:
    SERVER: ClassVar[str] = "index.commoncrawl.org"

    target_url: str

    agent: str = (
        "cc-get-started/1.0 (Example data retrieval script; feiyangc@example.com)"
    )

    def gen_index_records(self):
        for index in all_available_indexes()[:2]:
            yield from self.process_single_index(index["cdx-api"])

    def process_single_index(self, index_cdx: str):
        index_url = urlunparse(
            (
                "https",
                index_cdx.removeprefix("https://"),
                "",
                "",
                urlencode(dict(url=self.target_url, output="json")),
                "",
            )
        )

        res = requests.get(index_url, headers={"user-agent": self.agent})

        if res.status_code == 200:
            for line in res.iter_lines():
                yield json.loads(line)
        else:
            return

    def iter_pages(self, record: dict):
        offset = int(record["offset"])
        length = int(record["length"])
        filename = record["filename"]
        result = {}

        if record["status"] == "200":
            s3_url = f"https://data.commoncrawl.org/{filename}"

            # Define the byte range for the request
            byte_range = f"bytes={offset}-{offset + length - 1}"

            response = requests.get(
                s3_url,
                headers={"user-agent": self.agent, "Range": byte_range},
                stream=True,
            )
            if response.status_code == 206:
                stream = ArchiveIterator(response.raw)
                warc_record: ArcWarcRecord
                for warc_record in stream:
                    result[record["timestamp"]] = warc_record.content_stream().read()

        return result