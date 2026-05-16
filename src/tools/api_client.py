import httpx
from src.config import settings


class GaokaoAPIClient:
    def __init__(self):
        self.base_url = settings.backend_api_url
        self.timeout = settings.backend_timeout

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    async def rank_query(
        self, province: str, category: str, score: float, year: int = 2025
    ) -> dict:
        async with self._client() as client:
            resp = await client.get(
                "/rank_query",
                params={
                    "province": province,
                    "category": category,
                    "score": int(score),
                    "year": year,
                },
            )
            resp.raise_for_status()
            return resp.json()

    async def recommend(self, payload: dict) -> dict:
        async with self._client() as client:
            resp = await client.post("/recommend", json=payload)
            resp.raise_for_status()
            return resp.json()

    async def school_analysis(self, school_name: str) -> dict:
        async with self._client() as client:
            resp = await client.get(
                "/school_analysis", params={"school_name": school_name}
            )
            resp.raise_for_status()
            return resp.json()

    async def major_analysis(self, major_name: str) -> dict:
        async with self._client() as client:
            resp = await client.get(
                "/major_analysis", params={"major_name": major_name}
            )
            resp.raise_for_status()
            return resp.json()

    async def search(self, keyword: str, search_type: str = "all") -> dict:
        async with self._client() as client:
            resp = await client.get(
                "/search",
                params={"keyword": keyword, "search_type": search_type},
            )
            resp.raise_for_status()
            return resp.json()

    async def score_query(
        self, province: str, category: str, rank: int, year: int = 2025
    ) -> dict:
        """位次转分数"""
        async with self._client() as client:
            resp = await client.get(
                "/score_query",
                params={
                    "province": province,
                    "category": category,
                    "rank": rank,
                    "year": year,
                },
            )
            resp.raise_for_status()
            return resp.json()

    async def get_provinces(self) -> dict:
        async with self._client() as client:
            resp = await client.get("/provinces")
            resp.raise_for_status()
            return resp.json()

    async def get_school_tiers(self) -> dict:
        async with self._client() as client:
            resp = await client.get("/school_tiers")
            resp.raise_for_status()
            return resp.json()

    async def get_major_categories(self) -> dict:
        async with self._client() as client:
            resp = await client.get("/major_categories")
            resp.raise_for_status()
            return resp.json()


api_client = GaokaoAPIClient()
