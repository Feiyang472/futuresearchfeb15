import datetime
import json
import os
import random
import sys
import urllib
from dataclasses import dataclass
from functools import cache

# For parsing WARC records:
from pathlib import Path
from typing import ClassVar, NamedTuple
from urllib.error import URLError
from urllib.parse import quote, quote_plus, urlencode, urljoin, urlunparse
from urllib.request import Request, urlopen

import crawl4ai
import crawl4ai.html2text
import numpy as np
import openai
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import tqdm
import tqdm.autonotebook
from dotenv import load_dotenv
from pydantic import BaseModel
from warcio.archiveiterator import ArchiveIterator
from warcio.recordloader import ArcWarcRecord


@cache
def all_available_indexes():
    url = "https://index.commoncrawl.org/collinfo.json"
    response = requests.get(url)
    return response.json()


@dataclass
class CommonCrawlContent:
    SERVER: ClassVar[str] = "index.commoncrawl.org"

    target_url: str
    recent_num: int = 2

    agent: str = (
        "cc-get-started/1.0 (Example data retrieval script; feiyangc@example.com)"
    )

    def gen_index_records(self):
        for i, index in enumerate(all_available_indexes()[: self.recent_num]):
            if i == 0:
                result = list(self.process_single_index(index_cdx=index["cdx-api"]))
                if len(result) == 0:
                    raise ValueError(f"{self.target_url} not likely to be crawled.")
                yield from result
            yield from self.process_single_index(index["cdx-api"])

    @property
    def data_path(self):
        return Path("data") / self.target_url

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
                    timestamp: str = record["timestamp"] + ".html"
                    (self.data_path / timestamp).write_bytes(
                        warc_record.content_stream().read()
                    )

    def save_all(self):
        self.data_path.mkdir(exist_ok=True, parents=True)
        list(
            map(self.iter_pages, tqdm.autonotebook.tqdm(list(self.gen_index_records())))
        )

    def get_timestamp_and_pure_text(self):
        return [
            (
                datetime.datetime.strptime(path.stem, "%Y%m%d%H%M%S"),
                (
                    crawl4ai.html2text.html2text(
                        path.read_text(),
                    )
                ),
            )
            for path in self.data_path.glob("*.html")
        ]

    def gen_chat_response(self, field: str):
        result = dict[str, ResponseTemplate]()
        for time, policies in tqdm.autonotebook.tqdm(
            scraper.get_timestamp_and_pure_text()
        ):
            file = f"{field}_responses" / self.data_path / f"{time.isoformat()}.txt"
            (f"{field}_responses" / self.data_path).mkdir(parents=True, exist_ok=True)
            if file.is_file():
                result[time] = ResponseTemplate.model_validate(
                    json.loads(file.read_text())
                )
            else:
                result[time] = generate_feasibility(policies, field)
                file.write_text(result[time].model_dump_json())

        return result


# Load environment variables
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ResponseTemplate(BaseModel):
    value: float
    confidence_level: float
    summary_reason: str


def generate_feasibility(policy, field,):
    # Generate a number between +1 (very supportive) and -1 (very unsupportive) to indicate whether the following policy '{policy}' supports the start-up field '{field}'"
    prompt = f"Generate a score number between +1 (very supportive / favorable) and -1 (very unsupportive / unfavorable) to indicate whether the following policy (in this case, Innovate UK grant scheme) '{policy}' supports this particular start-up field '{field}'; your output is this number plus always a short rationale of less than 3 sentences. Also generate a confidence level of a number that is between 0 and 1 to describe how confident you are about the score."
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. You will have received grant info from Innovate UK, and a start-up field that your audience is planning to build a start-up in. Now, based on the number of available grants relevant to that field, and your observation on the relevance, scale, accessibility, and ease of application, your task is to assess how supportive / favorable is the grant scheme as a whole towards that particular industry or start-up field. A very supportive / favorable output would be: more than 2 relevant grant schemes that seem to be of a big scale, highly accessible, and easy to apply. A completely not supportive / favorable grant scheme would look like no relevant grant at all. You should be very specific and reasonable in your assessment; you should also give a rationale for why you give this particular score, along with your confidence level.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format=ResponseTemplate,
        max_tokens=200,
        n=1,
        temperature=0.7,
    )
    return response.choices[0].message.parsed


def generate_favorability_data(topic1, topic2, start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    topic1_favorability = np.random.randint(30, 71, size=len(date_range)) / 100
    topic2_favorability = np.random.randint(30, 71, size=len(date_range)) / 100

    df = pd.DataFrame(
        {"Date": date_range, topic1: topic1_favorability, topic2: topic2_favorability}
    )

    return df


if __name__ == "__main__":
    st.set_page_config(page_title="Topic Favorability Comparison", layout="wide")

    st.title("Topic Favorability Comparison Over Time")

    col1, col2 = st.columns(2)

    with col1:
        field = st.text_input("Enter the field", value="Fintech")

    with col2:
        policy_url = st.text_input(
            "Enter the policy url",
            value="https://iuk-business-connect.org.uk/opportunities",
        )

    # start_date = st.date_input("Start date", value=pd.to_datetime("2023-01-01"))
    # end_date = st.date_input("End date", value=pd.to_datetime("2023-12-31"))

    if st.button("Generate Favorability Data and Features"):
        with st.spinner("Generating features and data..."):
            scraper = CommonCrawlContent(policy_url, -1)

            structured_result = scraper.gen_chat_response(field=field)

            numerical_result = pd.Series(
                {k: v.value for k, v in structured_result.items()}
            )
            text_result = pd.DataFrame(
                [
                    {"date": k, "reason": v.summary_reason}
                    for k, v in structured_result.items()
                ]
            )
        st.line_chart(numerical_result)
        st.table(text_result.sort_values("date"))

    # if st.button("Generate Favorability Data and Features"):
    #     if start_date < end_date:
    #         with st.spinner("Generating features and data..."):
    #             # Generate features
    #             topic1_features = generate_features(topic1)
    #             topic2_features = generate_features(topic2)

    #             # Generate tag points
    #             tag_points = generate_tag_points(topic1, topic2)

    #             # Generate favorability data
    #             df = generate_favorability_data(topic1, topic2, start_date, end_date)

    #             # Display features
    #             st.subheader("Key Features")
    #             col1, col2 = st.columns(2)
    #             with col1:
    #                 st.write(f"{topic1} Features:")
    #                 for feature in topic1_features:
    #                     st.write(f"- {feature}")
    #             with col2:
    #                 st.write(f"{topic2} Features:")
    #                 for feature in topic2_features:
    #                     st.write(f"- {feature}")

    #             # Create the line chart
    #             fig = px.line(df, x='Date', y=[topic1, topic2], title=f"Favorability Comparison: {topic1} vs {topic2}")
    #             fig.update_layout(yaxis_title="Favorability", xaxis_title="Date")

    #             # Add annotation points
    #             num_annotations = min(len(tag_points), 5)
    #             annotation_dates = random.sample(df['Date'].tolist(), num_annotations)
    #             annotation_dates.sort()

    #             for i, date in enumerate(annotation_dates):
    #                 y_position = df.loc[df['Date'] == date, topic1].values[0]
    #                 fig.add_annotation(
    #                     x=date,
    #                     y=y_position,
    #                     text=f"Point {i+1}",
    #                     showarrow=True,
    #                     arrowhead=2,
    #                     arrowsize=1,
    #                     arrowwidth=2,
    #                     arrowcolor="#636363",
    #                     ax=0,
    #                     ay=-40
    #                 )

    #             st.plotly_chart(fig, use_container_width=True)

    #             # Display tag points
    #             st.subheader("Key Events/Quotes")
    #             for i, (date, tag) in enumerate(zip(annotation_dates, tag_points)):
    #                 st.write(f"**Point {i+1} ({date.strftime('%Y-%m-%d')}):** {tag}")

    #             st.write("Sample Data:")
    #             st.dataframe(df)
    #     else:
    #         st.error("Error: End date must be after the start date.")

    st.sidebar.header("About")
    st.sidebar.info(
        "This Streamlit app compares the favorability of two topics over time. "
        "Enter the topics you want to compare, select a date range, and click 'Generate Favorability Data and Features' "
        "to see a time series visualization of their relative favorability, along with key features and events."
    )
