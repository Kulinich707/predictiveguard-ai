import json
from contextlib import asynccontextmanager

import asyncpg

DDL = """
CREATE TABLE IF NOT EXISTS scoring_tasks (
 task_id UUID PRIMARY KEY,
 payload JSONB NOT NULL,
 status TEXT NOT NULL CHECK (status IN
 ('SUBMITTING','UNKNOWN','PENDING','RUNNING','SUCCEEDED','FAILED')),
 attempts INTEGER NOT NULL DEFAULT 0,
 result JSONB,
 error TEXT,
 created_at TIMESTAMPTZ NOT NULL,
 updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 inference_seconds DOUBLE PRECISION,
 delay_seconds DOUBLE PRECISION,
 total_seconds DOUBLE PRECISION
);
ALTER TABLE scoring_tasks
 ADD COLUMN IF NOT EXISTS trace_context JSONB,
 ADD COLUMN IF NOT EXISTS deliveries INTEGER NOT NULL DEFAULT 0,
 ADD COLUMN IF NOT EXISTS dlq_confirmed BOOLEAN NOT NULL DEFAULT FALSE;
"""


class TaskStore:
    def __init__(self, config):
        self.config = config
        self.pool = None

    async def start(self):
        self.pool = await asyncpg.create_pool(
            self.config.postgres_dsn,
            min_size=1,
            max_size=max(4, self.config.worker_concurrency + 2),
            command_timeout=60,
        )
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(hashtext('scoring_tasks schema'))")
            await conn.execute(DDL)

    async def close(self):
        if self.pool:
            await self.pool.close()

    @asynccontextmanager
    async def locked(self, task_id):
        async with self.pool.acquire() as conn:
            key = str(task_id)
            await conn.execute("SELECT pg_advisory_lock(hashtextextended($1, 0))", key)
            try:
                yield conn
            finally:
                await conn.execute("SELECT pg_advisory_unlock(hashtextextended($1, 0))", key)

    async def get(self, task_id, conn=None):
        if conn is None:
            async with self.pool.acquire() as connection:
                return await self.get(task_id, connection)
        row = await conn.fetchrow("SELECT * FROM scoring_tasks WHERE task_id=$1", task_id)
        if row is None:
            return None
        item = dict(row)
        for field in ("payload", "result", "trace_context"):
            if item[field] is not None:
                item[field] = json.loads(item[field])
        return item

    async def ensure(self, conn, task, status="PENDING", *, trace_context=None):
        context_json = json.dumps(trace_context) if trace_context else None
        await conn.execute(
            """INSERT INTO scoring_tasks(task_id,payload,status,created_at,trace_context)
            VALUES($1,$2::jsonb,$3,$4,$5::jsonb) ON CONFLICT DO NOTHING""",
            task.task_id,
            json.dumps(task.features.model_dump()),
            status,
            task.created_at,
            context_json,
        )
        row = await self.get(task.task_id, conn)
        if row["payload"] != task.features.model_dump():
            raise ValueError("task_id already belongs to different features")
        if row["trace_context"] is None and trace_context:
            await conn.execute(
                "UPDATE scoring_tasks SET trace_context=$2::jsonb WHERE task_id=$1",
                task.task_id,
                context_json,
            )
            row["trace_context"] = trace_context
        return row

    async def delivery(self, conn, task_id):
        return await conn.fetchval(
            """UPDATE scoring_tasks SET deliveries=deliveries+1
            WHERE task_id=$1 RETURNING deliveries""",
            task_id,
        )

    async def state(self, conn, task_id, status, error=None):
        await conn.execute(
            "UPDATE scoring_tasks SET status=$2,error=$3,updated_at=now() WHERE task_id=$1",
            task_id,
            status,
            error,
        )

    async def attempt(self, conn, task_id):
        return await conn.fetchval(
            """UPDATE scoring_tasks SET attempts=attempts+1,status='RUNNING',updated_at=now()
            WHERE task_id=$1 RETURNING attempts""",
            task_id,
        )

    async def success(self, conn, task, result, inference_seconds, delay_seconds):
        return await conn.fetchval(
            """UPDATE scoring_tasks SET status='SUCCEEDED',result=$2::jsonb,
            error=NULL,inference_seconds=$3,delay_seconds=$4,
            total_seconds=greatest(0,extract(epoch FROM (clock_timestamp()-created_at))),
            updated_at=now() WHERE task_id=$1 RETURNING total_seconds""",
            task.task_id,
            result.model_dump_json(),
            inference_seconds,
            delay_seconds,
        )

    async def failed_to_dlq(self, conn, task_id, error):
        await conn.execute(
            """UPDATE scoring_tasks SET status='FAILED',error=$2,dlq_confirmed=true,
            updated_at=now() WHERE task_id=$1""",
            task_id,
            error,
        )
