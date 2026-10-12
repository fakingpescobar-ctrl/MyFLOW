/**
 * MyFLOW → DeepSeek Harness: вкладывает надиктованный текст в текущую сессию.
 *
 * Зачем: у MyFLOW есть режим инжекта в чат OpenCode (через консольный буфер, без
 * фокуса). Для DSH этот путь не подходит (это десктопное приложение, а не консоль),
 * поэтому здесь штатный путь самого DSH — плагин, живущий внутри приложения и
 * вызывающий API агента:
 *
 *     agent.followup({ content: [{ type: 'text', text }], source: {...} })
 *
 * Сообщение становится обычной репликой пользователя в текущей сессии — ровно то,
 * что происходит, когда человек печатает в строке ввода и жмёт Enter.
 *
 * Связь — через файл-инбокс в домашнем каталоге DSH (никаких портов и токенов):
 *
 *     $DSH_HOME/myflow-inbox.jsonl   строки вида {"text": "...", "submit": true}
 *     $DSH_HOME/myflow-inbox.state   докуда уже прочитали (смещение в байтах)
 *
 * MyFLOW дописывает строку — плагин её забирает и вкладывает в сессию. Если
 * активной сессии нет, строка остаётся непрочитанной и уйдёт при следующем опросе.
 */
import { closeSync, existsSync, openSync, readFileSync, readSync, statSync,
         writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';

export const name = 'myflow-bridge';

// Плагину нужна служба агентов: через неё находим сессию и вкладываем реплику.
export const inject = ['agents'];

const POLL_MS = 250;
const PLUGIN_ID = 'myflow-bridge';

function dshHome() {
  return process.env.DSH_HOME || join(homedir(), '.dsh');
}

/** Запомненная сессия: после первой удачной вставки держимся её. */
let lastAgent = undefined;

/** Время последней активности агента (поле называется по-разному — пробуем всё). */
function activityStamp(agent) {
  try {
    const session = agent?.session ?? {};
    const values = [session.updatedAt, session.lastActivityAt, session.lastEventAt,
                    session.time, session.createdAt, agent?.updatedAt];
    for (const value of values) {
      const num = typeof value === 'number' ? value : Date.parse(value);
      if (Number.isFinite(num) && num > 0) return num;
    }
  } catch {
    /* нет доступа — считаем неизвестным */
  }
  return 0;
}

/**
 * Куда вкладывать: самая АКТИВНАЯ сессия, а не просто последняя в списке.
 *
 * Раньше брали «текущего инициатора, иначе последнего в списке» — при нескольких
 * открытых чатах диктовка уходила не в тот чат. Теперь по порядку:
 *   1) сессия, куда уже успешно вкладывали (держимся её);
 *   2) агент, инициирующий текущий ход;
 *   3) агент с самой поздней активностью;
 *   4) запасной вариант — последний в списке.
 */
function targetAgent(ctx) {
  try {
    const list = [...(ctx.agents.list?.() ?? []), ...(ctx.agents.roots?.() ?? [])];
    if (lastAgent && list.includes(lastAgent)) return lastAgent;

    const current = ctx.agents.currentInitiator?.();
    if (current) return current;

    if (list.length) {
      const byActivity = list.slice().sort((a, b) => activityStamp(b) - activityStamp(a));
      if (activityStamp(byActivity[0]) > 0) return byActivity[0];
      return list[list.length - 1];
    }
  } catch {
    /* нет службы или она не готова — попробуем в следующий раз */
  }
  return undefined;
}

/** Вложить реплику в сессию. true — ушло. */
function deliver(ctx, text) {
  const agent = targetAgent(ctx);
  if (!agent) return false;
  // ВАЖНО про источник сообщения. Диктовка — это ПРЯМАЯ реплика человека, поэтому
  // источник встроенный: { kind: 'user' }. Своя разновидность (kind: 'plugin')
  // требует, чтобы плагин её ЗАРЕГИСТРИРОВАЛ в типе сообщений; без регистрации
  // рантайм отвергает ход с ошибкой «format v4 message requires a producer-owned
  // source kind» — на этом мы уже спотыкались.
  const message = {
    id: `myflow-${Date.now()}-${randomUUID()}`,
    role: 'user',
    content: [{ type: 'text', text }],
    source: { kind: 'user' },
  };
  try {
    if (typeof agent.followup === 'function') {
      agent.followup(message);
      lastAgent = agent;      // дальше держимся этого чата
      return true;
    }
    if (typeof agent.send === 'function') {
      agent.send(message, 'next-turn', true);
      return true;
    }
    ctx.logger?.warn?.('myflow-bridge: у агента нет followup/send');
  } catch (error) {
    ctx.logger?.warn?.(`myflow-bridge: не удалось вложить сообщение: ${error}`);
  }
  return false;
}

export function apply(ctx) {
  const home = dshHome();
  const inbox = join(home, 'myflow-inbox.jsonl');
  const stateFile = join(home, 'myflow-inbox.state');

  let offset = 0;
  try {
    offset = Number.parseInt(readFileSync(stateFile, 'utf8').trim(), 10) || 0;
  } catch {
    offset = 0;
  }

  const drain = () => {
    let size = 0;
    try {
      size = statSync(inbox).size;
    } catch {
      return; // инбокса ещё нет — это норма
    }
    if (size < offset) offset = 0; // файл усекли (перезапуск MyFLOW)
    if (size === offset) return;

    let chunk = '';
    try {
      const fd = openSync(inbox, 'r');
      try {
        const length = size - offset;
        const buffer = Buffer.alloc(length);
        readSync(fd, buffer, 0, length, offset);
        chunk = buffer.toString('utf8');
      } finally {
        closeSync(fd);
      }
    } catch (error) {
      ctx.logger?.warn?.(`myflow-bridge: чтение инбокса: ${error}`);
      return;
    }

    // Обрабатываем только ЦЕЛЫЕ строки: последняя может дописываться прямо сейчас.
    const lastNewline = chunk.lastIndexOf('\n');
    if (lastNewline < 0) return;
    const complete = chunk.slice(0, lastNewline);
    const consumed = offset + Buffer.byteLength(complete, 'utf8') + 1;

    for (const line of complete.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      let item;
      try {
        item = JSON.parse(trimmed);
      } catch {
        ctx.logger?.warn?.('myflow-bridge: строка инбокса не разобрана как JSON');
        continue;
      }
      const text = typeof item?.text === 'string' ? item.text.trim() : '';
      if (!text) continue;
      if (deliver(ctx, text)) {
        ctx.logger?.info?.(`myflow-bridge: вложено ${text.length} символов`);
      } else {
        // Нет активной сессии — оставляем строку на следующий опрос.
        ctx.logger?.info?.('myflow-bridge: нет активной сессии, жду');
        return;
      }
    }

    offset = consumed;
    try {
      writeFileSync(stateFile, String(offset), 'utf8');
    } catch (error) {
      ctx.logger?.warn?.(`myflow-bridge: запись состояния: ${error}`);
    }
  };

  const timer = setInterval(drain, POLL_MS);
  timer.unref?.();
  ctx.on?.('dispose', () => clearInterval(timer));
  drain();

  ctx.logger?.info?.(`myflow-bridge: слушаю ${inbox}`);
}
