"""Admin CLI for managing users and the database."""

import asyncio
import json
import secrets
import string
from types import TracebackType
from typing import Optional, Type

import typer
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.operations import SearchIndexModel

from whyhow_api.config import Settings
from whyhow_api.database import (
    close_mongo_connection,
    connect_to_mongo,
    get_client,
)

app = typer.Typer()


class MongoDBConnection:
    """Context manager for MongoDB connection."""

    def __init__(self) -> None:
        self.settings = Settings()
        self.client = None
        self.db = None

    def __enter__(self) -> AsyncIOMotorDatabase:
        """Connect to MongoDB and return database."""
        connect_to_mongo(uri=self.settings.mongodb.uri)
        self.client = get_client()  # type: ignore[assignment]
        if self.client is None:
            raise ConnectionError("Failed to connect to MongoDB client.")
        self.db = self.client.get_database(self.settings.mongodb.database_name)
        return self.db

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Close the MongoDB connection."""
        close_mongo_connection()


def generate_api_key(length: int = 40) -> str:
    """Generate a random API key."""
    characters = string.ascii_letters + string.digits
    return "".join(secrets.choice(characters) for _ in range(length))


async def setup_collections_and_indexes(
    db: AsyncIOMotorDatabase, config_file: str
) -> None:
    """Set up collections and indexes from a configuration file."""
    with open(config_file, "r") as file:
        config = json.load(file)

    for collection_name, details in config.items():
        # Create collection if it doesn't exist
        existing_collections = await db.list_collection_names()
        if collection_name not in existing_collections:
            await db.create_collection(collection_name)
            print(f"Created collection: {collection_name}")
        else:
            print(f"Collection {collection_name} already exists.")

        # Create regular indexes
        for index in details.get("regular_indexes", []):
            try:
                key = [(field, direction) for field, direction in index["key"]]
                index_options = {
                    k: v for k, v in index.items() if k not in ["key", "name"]
                }
                await db[collection_name].create_index(
                    key, name=index["name"], **index_options
                )
                print(
                    f"Created regular index '{index['name']}' on collection '{collection_name}'."
                )
            except Exception as e:
                print(
                    f"Failed to create regular index '{index['name']}' on collection '{collection_name}': {e}"
                )

        # Create search indexes
        for search_index in details.get("search_indexes", []):
            try:
                if search_index["type"] == "search":
                    # For Atlas Search indexes
                    definition = {
                        "mappings": {
                            "dynamic": True  # Changed to True since fields array is empty
                        }
                    }
                    await db[collection_name].create_search_index(
                        {
                            "name": search_index["name"],
                            "definition": definition,
                        }
                    )
                elif search_index["type"] == "vectorSearch":
                    # For Vector Search indexes
                    search_index_model = SearchIndexModel(
                        definition={"fields": search_index["fields"]},
                        name=search_index["name"],
                        type="vectorSearch",
                    )
                    await db[collection_name].create_search_index(
                        model=search_index_model
                    )

                print(
                    f"Created {search_index['type']} index '{search_index['name']}' on collection '{collection_name}'."
                )
            except Exception as e:
                print(
                    f"Failed to create search index '{search_index['name']}' on collection '{collection_name}': {str(e)}"
                )


async def create_user_in_db(
    db: AsyncIOMotorDatabase, email: str, openai_key: str
) -> None:
    """Add a user with an API key to the database."""
    api_key = generate_api_key()
    user = {
        "email": email,
        "api_key": api_key,
        "providers": [
            {
                "type": "llm",
                "value": "byo-openai",
                "api_key": openai_key,
                "metadata": {
                    "byo-openai": {
                        "language_model_name": "gpt-4o",
                        "embedding_name": "text-embedding-3-small",
                    }
                },
            }
        ],
    }
    await db["user"].insert_one(user)
    print(f"User created with email: {email}, API Key: {api_key}")


@app.command()
def setup_collections(
    config_file: str = typer.Option(
        ..., help="Path to the configuration JSON file."
    )
) -> None:
    """Set up collections and indexes."""
    with MongoDBConnection() as db:
        asyncio.run(setup_collections_and_indexes(db, config_file))


@app.command()
def create_user(
    email: str = typer.Option(..., help="Email address of the user."),
    openai_key: str = typer.Option(
        ..., help="OpenAI key associated with the user."
    ),
) -> None:
    """Create a user with an API key."""
    with MongoDBConnection() as db:
        asyncio.run(create_user_in_db(db, email, openai_key))


if __name__ == "__main__":
    app()
