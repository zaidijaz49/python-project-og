import { useEffect, useState } from "react";
import { api } from "../api/client";

// "checking" -> "online" | "offline". Checked when the chat opens.
export function useServerStatus(active) {
  const [status, setStatus] = useState("checking");

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    api
      .health()
      .then(() => !cancelled && setStatus("online"))
      .catch(() => !cancelled && setStatus("offline"));
    return () => {
      cancelled = true;
    };
  }, [active]);

  return status;
}
