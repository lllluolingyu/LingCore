// Guard attached pages without keeping their sockets dependent on Playwright.
({ binding, key, token }) => {
  if (globalThis[key]) return;
  const NativeWebSocket = globalThis.WebSocket;
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "WebSocket");
  const pending = new Set();
  let active = true;
  const state = {
    restore(secret) {
      if (secret !== token) return;
      active = false;
      if (globalThis.WebSocket === WebSocket) {
        Object.defineProperty(globalThis, "WebSocket", descriptor);
      }
      // Decisions still in flight must not depend on a disconnected binding.
      for (const connect of [...pending]) connect(true);
      delete globalThis[key];
    },
  };

  class WebSocket extends EventTarget {
    #socket;
    #url;
    #protocols;
    #readyState = NativeWebSocket.CONNECTING;
    #binaryType = "blob";
    #handlers = new Map();

    constructor(url, protocols) {
      super();
      const parsed = new URL(url, document.baseURI);
      if (parsed.protocol === "http:") parsed.protocol = "ws:";
      if (parsed.protocol === "https:") parsed.protocol = "wss:";
      if (!["ws:", "wss:"].includes(parsed.protocol) || parsed.href.includes("#")) {
        throw new DOMException("Invalid WebSocket URL", "SyntaxError");
      }
      const list = typeof protocols === "string" ? [protocols] : Array.from(protocols ?? [], String);
      if (new Set(list).size !== list.length || list.some(value => !/^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$/.test(value))) {
        throw new DOMException("Invalid WebSocket protocols", "SyntaxError");
      }
      this.#url = parsed.href;
      this.#protocols = list;
      const connect = allowed => {
        if (!pending.delete(connect) || this.#readyState !== NativeWebSocket.CONNECTING) return;
        if (!allowed) {
          this.#finish(1008, "Blocked by the browser network policy", false);
          return;
        }
        try {
          const socket = this.#socket = new NativeWebSocket(this.#url, this.#protocols);
          socket.binaryType = this.#binaryType;
          socket.addEventListener("open", () => this.dispatchEvent(new Event("open")));
          socket.addEventListener("error", () => this.dispatchEvent(new Event("error")));
          socket.addEventListener("message", event => this.dispatchEvent(new MessageEvent("message", {
            data: event.data, origin: event.origin, lastEventId: event.lastEventId,
          })));
          socket.addEventListener("close", event => this.dispatchEvent(new CloseEvent("close", {
            code: event.code, reason: event.reason, wasClean: event.wasClean,
          })));
        } catch {
          this.#finish(1006, "", false);
        }
      };
      pending.add(connect);
      Promise.resolve().then(() => active ? globalThis[binding](this.#url) : true)
        .then(allowed => connect(!active || allowed === true), () => connect(!active));
    }

    #finish(code, reason, wasClean) {
      this.#readyState = NativeWebSocket.CLOSED;
      queueMicrotask(() => {
        this.dispatchEvent(new Event("error"));
        this.dispatchEvent(new CloseEvent("close", { code, reason, wasClean }));
      });
    }

    get url() { return this.#url; }
    get readyState() { return this.#socket?.readyState ?? this.#readyState; }
    get protocol() { return this.#socket?.protocol ?? ""; }
    get extensions() { return this.#socket?.extensions ?? ""; }
    get bufferedAmount() { return this.#socket?.bufferedAmount ?? 0; }
    get binaryType() { return this.#socket?.binaryType ?? this.#binaryType; }
    set binaryType(value) {
      if (value !== "blob" && value !== "arraybuffer") return;
      this.#binaryType = value;
      if (this.#socket) this.#socket.binaryType = value;
    }

    send(data) {
      if (this.readyState === NativeWebSocket.CONNECTING) {
        throw new DOMException("WebSocket is still connecting", "InvalidStateError");
      }
      if (this.#socket) this.#socket.send(data);
    }

    close(code, reason = "") {
      if (this.#socket) {
        this.#socket.close(...arguments);
        return;
      }
      if (code !== undefined && code !== 1000 && !(code >= 3000 && code <= 4999)) {
        throw new DOMException("Invalid WebSocket close code", "InvalidAccessError");
      }
      reason = String(reason);
      if (new TextEncoder().encode(reason).length > 123) {
        throw new DOMException("WebSocket close reason is too long", "SyntaxError");
      }
      if (this.#readyState === NativeWebSocket.CONNECTING) this.#finish(1006, "", false);
    }

    static [Symbol.hasInstance](value) {
      return NativeWebSocket.prototype.isPrototypeOf(value) || this.prototype.isPrototypeOf(value);
    }

    get [Symbol.toStringTag]() { return "WebSocket"; }

    _getHandler(type) { return this.#handlers.get(type) ?? null; }
    _setHandler(type, handler) {
      const previous = this.#handlers.get(type);
      if (previous) this.removeEventListener(type, previous);
      this.#handlers.delete(type);
      if (typeof handler === "function") {
        this.#handlers.set(type, handler);
        this.addEventListener(type, handler);
      }
    }
  }

  for (const name of ["CONNECTING", "OPEN", "CLOSING", "CLOSED"]) {
    Object.defineProperty(WebSocket, name, { value: NativeWebSocket[name], enumerable: true });
    Object.defineProperty(WebSocket.prototype, name, { value: NativeWebSocket[name], enumerable: true });
  }
  for (const type of ["open", "error", "message", "close"]) {
    Object.defineProperty(WebSocket.prototype, "on" + type, {
      configurable: true, enumerable: true,
      get() { return this._getHandler(type); },
      set(handler) { this._setHandler(type, handler); },
    });
  }
  Object.defineProperty(globalThis, key, { value: state, configurable: true });
  Object.defineProperty(globalThis, "WebSocket", { ...descriptor, value: WebSocket });
}
