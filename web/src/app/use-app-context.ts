import { useOutletContext } from "react-router-dom";

import type { AppOutletContext } from "./app-context";

export function useAppContext() {
  return useOutletContext<AppOutletContext>();
}
