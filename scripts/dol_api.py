"""Small, safe client for the DOL v4 OSHA data API."""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Callable, Iterator
from typing import Any

import requests
from dotenv import load_dotenv


BASE_URL = "https://apiprod.dol.gov/v4/get/OSHA"


class DOLApiError(RuntimeError):
    """A request failed; this is never used for a valid empty result."""


class DOLApiClient:
    def __init__(
        self,
        *,
        timeout_seconds: int = 60,
        max_retries: int = 4,
        logger: Callable[[str], None] = print,
        session: requests.Session | None = None,
    ) -> None:
        load_dotenv(".env")
        self.api_key = os.getenv("DOL_API_KEY", "").strip()
        if not self.api_key:
            raise DOLApiError("DOL_API_KEY could not be loaded from .env.")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.logger = logger
        self.session = session or requests.Session()

    def _log(
        self,
        endpoint: str,
        status: int | str,
        attempt: int,
        row_count: int | str,
        wait_seconds: float = 0.0,
    ) -> None:
        # Deliberately omit URLs, parameters, response bodies, and credentials.
        self.logger(
            f"DOL endpoint={endpoint} status={status} attempt={attempt} "
            f"rows={row_count} wait={wait_seconds:.1f}s"
        )

    @staticmethod
    def _records(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            data = payload
        elif isinstance(payload, dict):
            data = payload.get("data", [])
        else:
            raise DOLApiError("DOL API returned an unexpected JSON structure.")
        if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
            raise DOLApiError("DOL API JSON did not contain a record list.")
        return data

    @staticmethod
    def _retry_after(response: requests.Response, attempt: int) -> float:
        value = response.headers.get("Retry-After")
        if value:
            try:
                return max(0.0, float(value))
            except ValueError:
                pass
        return min(30.0, 2 ** (attempt - 1)) + random.uniform(0.0, 0.5)

    def get_records(
        self,
        endpoint: str,
        *,
        fields: list[str] | None = None,
        filter_object: dict[str, Any] | None = None,
        sort_by: str | None = None,
        sort: str | None = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Fetch one page. A 204 response is a successful zero-row page."""
        if limit < 1 or offset < 0:
            raise ValueError("limit must be positive and offset cannot be negative.")

        params: dict[str, str | int] = {
            "X-API-KEY": self.api_key,
            "limit": limit,
            "offset": offset,
        }
        if fields:
            params["fields"] = ",".join(fields)
        if filter_object:
            encoded_filter = json.dumps(filter_object, separators=(",", ":"))
            # Keeping filters modest prevents known 403 failures from long URLs.
            if len(encoded_filter) > 6_000:
                raise DOLApiError("DOL API filter is too large; split it into smaller batches.")
            params["filter_object"] = encoded_filter
        if sort_by:
            params["sort_by"] = sort_by
        if sort:
            params["sort"] = sort

        url = f"{BASE_URL}/{endpoint}/json"
        last_status: int | str = "request_error"
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout_seconds)
            except requests.RequestException:
                last_status = "request_error"
                if attempt == self.max_retries:
                    break
                wait = min(30.0, 2 ** (attempt - 1)) + random.uniform(0.0, 0.5)
                self._log(endpoint, last_status, attempt, "unknown", wait)
                time.sleep(wait)
                continue

            last_status = response.status_code
            if response.status_code == 204:
                self._log(endpoint, 204, attempt, 0)
                return []
            if response.status_code == 200:
                try:
                    rows = self._records(response.json()) if response.content.strip() else []
                except (ValueError, DOLApiError) as error:
                    self._log(endpoint, 200, attempt, "unknown")
                    raise DOLApiError("DOL API returned invalid JSON data.") from error
                self._log(endpoint, 200, attempt, len(rows))
                return rows

            retryable = response.status_code == 429 or 500 <= response.status_code < 600
            if not retryable or attempt == self.max_retries:
                self._log(endpoint, response.status_code, attempt, "unknown")
                break
            wait = self._retry_after(response, attempt)
            self._log(endpoint, response.status_code, attempt, "unknown", wait)
            time.sleep(wait)

        raise DOLApiError(
            f"DOL {endpoint} request failed after {self.max_retries} attempts (last status {last_status})."
        )

    def iter_pages(self, endpoint: str, **kwargs: Any) -> Iterator[list[dict[str, Any]]]:
        """Yield pages until a short or empty page marks the end of a result set."""
        limit = int(kwargs.pop("limit", 500))
        offset = int(kwargs.pop("offset", 0))
        while True:
            rows = self.get_records(endpoint, limit=limit, offset=offset, **kwargs)
            yield rows
            if len(rows) < limit:
                return
            offset += limit
