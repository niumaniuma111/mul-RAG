"""
业务服务层。

分层约定
--------
    api/          HTTP 边界：鉴权、参数校验、响应包装。**不含业务逻辑**
    services/     业务逻辑：编排、事务、外部依赖调用
    tasks/        Celery 任务：长耗时流程（入库），可重试
    db/           数据访问：ORM 模型与会话
    core/         横切关注点：配置、日志、锁、鉴权原语

⚠️ 为什么把编排放在 services 而不是 api 里：
   同一个流程既要被 HTTP 触发（上传），又要被 Celery 触发（解析），
   放在 api 层会导致 Celery 反向依赖 HTTP 层。services 是两者的公共下游。
"""

__all__: list[str] = []
