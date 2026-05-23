import { useEffect, useState } from "react";
import { api } from "../lib/api";

interface Health {
  status: string;
  cluster_status?: string;
  cluster_name?: string;
}

export function ClusterStatus() {
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    let mounted = true;
    const tick = async () => {
      try {
        const h = await api.health();
        if (mounted) setHealth(h);
      } catch {
        if (mounted) setHealth({ status: "error" });
      }
    };
    tick();
    const id = setInterval(tick, 15_000);
    return () => { mounted = false; clearInterval(id); };
  }, []);

  if (!health) return null;

  const ok = health.status === "ok" && (health.cluster_status === "green" || health.cluster_status === "yellow");
  const color = ok ? "bg-elastic-teal" : "bg-elastic-pink";

  return (
    <div className="flex items-center gap-2 text-[10px] font-mono text-elastic-gray uppercase tracking-widest">
      <span className={`w-1.5 h-1.5 rounded-full ${color} animate-pulse`} />
      <span>
        {ok ? `cluster ${health.cluster_status}` : "cluster offline"}
      </span>
    </div>
  );
}
