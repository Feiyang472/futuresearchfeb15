from dataclasses import dataclass
from typing import ClassVar, NamedTuple
from urllib.error import URLError
from urllib.request import Request, urlopen
import requests
import json

# For parsing URLs:
from urllib.parse import quote, quote_plus, urljoin, urlencode, urlunparse

# For parsing WARC records:
import urllib
from warcio.archiveiterator import ArchiveIterator
from warcio.recordloader import ArcWarcRecord


class StatusError(URLError): ...


@dataclass
class CommonCrawlContent:
    SERVER: ClassVar[str] = "index.commoncrawl.org"

    target_url: str
    index_name: str = "CC-MAIN-2024-51-index"

    agent: str = (
        "cc-get-started/1.0 (Example data retrieval script; yourname@example.com)"
    )

    def gen_index_records(self):
        index_url = urlunparse(
            (
                "https",
                self.SERVER,
                self.index_name,
                "",
                urlencode(dict(url=self.target_url, output="json")),
                "",
            )
        )
        print(index_url)

        res = requests.get(index_url, headers={"user-agent": self.agent})



        if res.status_code == 200:
            for line in res.iter_lines():
                yield json.loads(line)
        else:
            raise StatusError(res.status_code)

    def iter_pages(self, record: dict):
        offset = int(record["offset"])
        length = int(record["length"])
        filename = record["filename"]

        if record["status"] == "200":
            s3_url = f"https://data.commoncrawl.org/{filename}"

            # Define the byte range for the request
            byte_range = f"bytes={offset}-{offset+length-1}"

            response = requests.get(
                s3_url,
                headers={"user-agent": self.agent, "Range": byte_range},
                stream=True,
            )
            if response.status_code == 206:

                stream = ArchiveIterator(response.raw)
                warc_record: ArcWarcRecord
                print(record)
                for warc_