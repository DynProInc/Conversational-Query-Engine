from models.database import SessionLocal, QueryHistory, SavedQuery
from datetime import datetime

def save_query_to_history(
    user_id: str,
    prompt: str,
    sql_query: str = None,
    database: str = "default",
    model: str = "unknown",
    query_executed: bool = False,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    input_cost: float = 0.0,
    output_cost: float = 0.0,
    total_cost: float = 0.0
):
    """Save a query to the query history"""
    db = SessionLocal()
    try:
        query_history = QueryHistory(
            user_id=user_id,
            prompt=prompt,
            sql_query=sql_query,
            database=database,
            model=model,
            query_executed=query_executed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost
        )
        db.add(query_history)
        db.commit()
        return query_history.id
    except Exception as e:
        db.rollback()
        print(f"Error saving query to history: {e}")
        return None
    finally:
        db.close()

def save_query_to_saved_queries(
    user_id: str,
    prompt: str,
    name: str,
    description: str = "",
    sql_query: str = None,
    database: str = "default",
    tags: list = None,
    model: str = "unknown",
    query_executed: bool = False,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    input_cost: float = 0.0,
    output_cost: float = 0.0,
    total_cost: float = 0.0
):
    """Save a query to the saved queries"""
    db = SessionLocal()
    try:
        saved_query = SavedQuery(
            user_id=user_id,
            prompt=prompt,
            name=name,
            description=description,
            sql_query=sql_query,
            database=database,
            tags=tags or [],
            model=model,
            query_executed=query_executed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost
        )
        db.add(saved_query)
        db.commit()
        return saved_query.id
    except Exception as e:
        db.rollback()
        print(f"Error saving query to saved queries: {e}")
        return None
    finally:
        db.close() 