import { useEffect, useState } from "react";

type Setter<T> = (value: T | ((current: T) => T)) => void;

export function useStoredState<T>(key: string, initialValue: T): [T, Setter<T>] {
  const [value, setValue] = useState<T>(() => {
    const raw = window.localStorage.getItem(key);
    if (!raw) {
      return initialValue;
    }
    try {
      return JSON.parse(raw) as T;
    } catch {
      return initialValue;
    }
  });

  useEffect(() => {
    window.localStorage.setItem(key, JSON.stringify(value));
  }, [key, value]);

  const updateValue: Setter<T> = (nextValue) => {
    setValue((current) =>
      typeof nextValue === "function"
        ? (nextValue as (current: T) => T)(current)
        : nextValue
    );
  };

  return [value, updateValue];
}
