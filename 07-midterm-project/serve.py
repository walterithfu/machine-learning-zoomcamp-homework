from __future__ import annotations

import os
import uvicorn

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "9696"))
    uvicorn.run("predict:app", host=host, port=port, reload=False)
