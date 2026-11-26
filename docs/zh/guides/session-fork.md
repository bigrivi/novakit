# 会话分叉

`novakit/session/fork.py` 将一个会话及其整消息前缀复制到
新会话。源会话从不被修改。宿主实现窄 `SessionForkStore`
协议；`novakit` 拥有复制算法。

## 分叉会话

```python
from novakit.session.fork import fork_session

result = await fork_session(store, session_id, message_id=None)
print(result["session_id"], result["messages"])
```

参数：

- `store`——实现 `SessionForkStore` 的宿主持久化（`get_session`、
  `get_messages`、`save_session`、`set_session_route`、`add_message`）。
- `session_id`——要复制的源会话。
- `message_id`——包含的最后一条消息；省略则复制全部可见历史。
- `title`——副本标题，省略时为 `"<origin title> (fork)"`
  （后缀常量 `FORK_SUFFIX = " (fork)"`）。
- `default_agent_key`——源行无智能体键时的兜底所有者
  （默认为 `novakit.constants.DEFAULT_AGENT_KEY`）。

结果描述该副本：`session_id`、`title`、消息数、
`forked_from` 和 `fork_point_message_id`。

## 复制语义

- **整消息前缀**——`_messages_up_to` 返回截至 `message_id`（含）的消息；
  分叉从不在历史中间切片。未知的 `message_id` 抛出 `ForkPointNotFound`；
  缺失的源抛出 `FileNotFoundError`。
- **工具配对保留**——每条消息都带完整 `tool_calls` / `tool_call_id`、
  令牌计数、提供商元数据、推理字段和时间戳重新添加，
  因此工具调用/结果对保持完整。
- **元数据与路由复制**——分叉通过 `set_session_route` 继承工作区、项目、
  提供商、模型和推理强度，记录 `forked_from` / `fork_point_message_id`，
  并从复制的用户消息设置 `turn_count`。

## API 参考

::: novakit.session.fork.fork_session
::: novakit.session.fork.SessionForkStore
::: novakit.session.fork.ForkPointNotFound
