import asyncio, random, time

async def search_web(query: str) -> str:
  await asyncio.sleep(1.0)
  return f"web results for '{query}'"

async def query_db(sql: str) -> str:
  await asyncio.sleep(1.5)
  return f"db rows for '{sql}'"

async def flaky_api(x: int) -> str:
  await asyncio.sleep(0.5)
  if random.random() < 0.5:
    raise ConnectionError("upstream API failed")
  return f"api ok {x}"

async def sequential():
  start = time.perf_counter()
  a = await search_web("langgraph")
  b = await query_db("select * from tickets")
  print(f"sequential: {time.perf_counter() - start:.1f}s ->", a, "|", b)

async def parallel():
  start = time.perf_counter()
  a, b = await asyncio.gather(
    search_web("langgraph"),
    query_db("select * from tickets")
  )
  print(f"parallel:   {time.perf_counter() - start:.1f}s ->", a, "|", b)

async def with_timeout():
    try:
        async with asyncio.timeout(1.2):
            await query_db("slow query")
    except TimeoutError:
        print("timeout:    query_db took longer than 1.2s, giving up")

async def survive_failures():
    results = await asyncio.gather(
        *(flaky_api(i) for i in range(4)),
        return_exceptions=True,              # failures come back as values
    )
    for i, r in enumerate(results):
        status = "FAILED" if isinstance(r, Exception) else "ok"
        print(f"call {i}: {status} -> {r}")

async def main():
    await sequential()
    await parallel()
    await with_timeout()
    await survive_failures()


asyncio.run(main())