"""Middleware for FastAPI."""

import collections
import logging
from datetime import datetime, timezone

###########################
# Auth0 used for UI #######
###########################
# import jwt
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

# from fastapi.security import OAuth2AuthorizationCodeBearer
from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)
from starlette.responses import Response

from whyhow_api.config import Settings
from whyhow_api.dependencies import get_settings
from whyhow_api.utilities.routers import clean_url

logger = logging.getLogger(__name__)


# Initialize token_buckets with a lambda that sets initial tokens based on settings
def create_initial_bucket() -> dict[str, float]:
    """Create an initial token bucket."""
    settings: Settings = get_settings()
    return {
        "last_check": datetime.now(timezone.utc).timestamp(),
        "tokens": settings.api.bucket_capacity,  # Set initial tokens to the bucket capacity
    }


# This will store the token buckets for each user
token_buckets: dict[str, dict[str, float]] = collections.defaultdict(
    create_initial_bucket
)


class RateLimiter(BaseHTTPMiddleware):
    """Token bucket rate limiter middleware for FastAPI."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Dispatch the request."""
        settings: Settings = get_settings()
        path_pattern = clean_url(request.url.path)
        excluded_path = (
            path_pattern
            in settings.api.excluded_paths + settings.api.public_paths
        )  # Exclude paths from rate limiting (TODO: public paths should be rate limited on IP.)

        if excluded_path:
            return await call_next(request)

        user_key = await self.get_rate_limit_key(request, settings)
        rate = settings.api.limit_frequency_value  # Tokens added per second
        capacity = settings.api.bucket_capacity
        now = datetime.now(timezone.utc).timestamp()

        bucket = token_buckets[user_key]
        time_passed = max(
            0, now - bucket["last_check"]
        )  # Ensure non-negative time passed
        bucket["last_check"] = now

        # Add tokens to the bucket based on elapsed time
        tokens_to_add = time_passed * rate  # Tokens are added per second
        bucket["tokens"] = min(capacity, bucket["tokens"] + tokens_to_add)

        # Logging state before processing the request
        # logger.info(
        #     f"Processing request from {user_key}. Available tokens before request: {bucket['tokens']}. Tokens to add: {tokens_to_add}. Time passed: {time_passed}s."
        # )

        response: Response
        if bucket["tokens"] < 1:
            # logger.warning(
            #     f"Rate limit exceeded for user {user_key}. No tokens available."
            # )
            response = JSONResponse(
                content={"error": "Rate limit exceeded"}, status_code=429
            )
        else:
            bucket["tokens"] -= 1
            response = await call_next(request)
            # logger.info(
            #     f"Token deducted for user {user_key}. Tokens remaining: {bucket['tokens']}."
            # )

        # Update response headers for client info
        response.headers["X-RateLimit-Limit"] = str(capacity)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, int(bucket["tokens"]))
        )
        response.headers["X-RateLimit-Reset"] = str(
            int(now + (1 - bucket["tokens"]) * (1 / rate))
        )

        return response

    @staticmethod
    async def get_rate_limit_key(request: Request, settings: Settings) -> str:
        """Get the rate limit key."""
        try:
            api_key = request.headers.get("x-api-key")
            ###########################
            # Auth0 used for UI #######
            ###########################
            # oauth2_scheme = OAuth2AuthorizationCodeBearer(
            #     authorizationUrl=settings.api.auth0.authorize_url,
            #     tokenUrl=settings.api.auth0.token_url,
            #     auto_error=False,
            # )
            # token = await oauth2_scheme(request)
            if api_key:
                return api_key
            # elif token:
            #     if (
            #         settings.api.auth0.domain is None
            #         or settings.api.auth0.audience is None
            #         or settings.api.auth0.algorithm is None
            #     ):
            #         raise ValueError(
            #             "Auth0 domain, audience, and algorithm required"
            #         )
            #     domain = settings.api.auth0.domain.get_secret_value()
            #     audience = settings.api.auth0.audience.get_secret_value()
            #     algorithm = settings.api.auth0.algorithm

            #     signing_key = (
            #         request.app.state.jwks_client.get_signing_key_from_jwt(
            #             token
            #         ).key
            #     )

            #     payload = jwt.decode(
            #         token,
            #         signing_key,
            #         algorithms=[algorithm],
            #         audience=audience,
            #         issuer=f"https://{domain}/",
            #     )
            #     return payload["sub"]
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key or token is required",
                )
        except HTTPException as http_e:
            raise http_e
        except Exception as e:
            logger.error(
                f"Error getting rate limit key (unable to authorize): {e}"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unable to authorize",
            )
