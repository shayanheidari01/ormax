# ormax/expressions.py
"""
SQL Expressions and F() objects for Ormax ORM.
Supports atomic column arithmetic operations like:
UPDATE table SET balance = balance - 10
"""
from typing import Any, Tuple, List, Union, Optional


class Expression:
    """Base class for all SQL expressions."""

    def resolve_expression(
        self,
        query=None,
        allow_joins: bool = True,
        reuse=None,
        summarize: bool = False,
        placeholder: str = "?",
        **kwargs,
    ) -> Tuple[str, List[Any]]:
        """
        Compile the expression into SQL snippet and param list.
        Must return (sql_string, params_list)
        """
        raise NotImplementedError

    def __add__(self, other) -> "BinaryExpression":
        return BinaryExpression(self, "+", other)

    def __radd__(self, other) -> "BinaryExpression":
        return BinaryExpression(other, "+", self)

    def __sub__(self, other) -> "BinaryExpression":
        return BinaryExpression(self, "-", other)

    def __rsub__(self, other) -> "BinaryExpression":
        return BinaryExpression(other, "-", self)

    def __mul__(self, other) -> "BinaryExpression":
        return BinaryExpression(self, "*", other)

    def __rmul__(self, other) -> "BinaryExpression":
        return BinaryExpression(other, "*", self)

    def __truediv__(self, other) -> "BinaryExpression":
        return BinaryExpression(self, "/", other)

    def __rtruediv__(self, other) -> "BinaryExpression":
        return BinaryExpression(other, "/", self)

    def __floordiv__(self, other) -> "BinaryExpression":
        return BinaryExpression(self, "/", other)

    def __rfloordiv__(self, other) -> "BinaryExpression":
        return BinaryExpression(other, "/", self)

    def __mod__(self, other) -> "BinaryExpression":
        return BinaryExpression(self, "%", other)

    def __rmod__(self, other) -> "BinaryExpression":
        return BinaryExpression(other, "%", self)

    def __neg__(self) -> "UnaryExpression":
        return UnaryExpression("-", self)

    def __pos__(self) -> "UnaryExpression":
        return UnaryExpression("+", self)


class F(Expression):
    """
    An object that represents the value of a field or transformed field.
    Allows column arithmetic operations in updates, filters, and annotations.

    Example:
        from ormax import F

        # Subtract 10 from balance
        await User.objects().filter(id=1).update(balance=F('balance') - 10)

        # Add bonus to balance
        await User.objects().filter(id=1).update(balance=F('balance') + F('bonus'))
    """

    def __init__(self, name: str):
        self.name = name

    def resolve_expression(
        self,
        query=None,
        allow_joins: bool = True,
        reuse=None,
        summarize: bool = False,
        placeholder: str = "?",
        **kwargs,
    ) -> Tuple[str, List[Any]]:
        return self.name, []

    def __repr__(self):
        return f"F({self.name!r})"


class Value(Expression):
    """Wraps a raw Python value as an expression."""

    def __init__(self, value: Any):
        self.value = value

    def resolve_expression(
        self,
        query=None,
        allow_joins: bool = True,
        reuse=None,
        summarize: bool = False,
        placeholder: str = "?",
        **kwargs,
    ) -> Tuple[str, List[Any]]:
        return placeholder, [self.value]

    def __repr__(self):
        return f"Value({self.value!r})"


class BinaryExpression(Expression):
    """Combines two expressions with a binary operator (+, -, *, /, etc.)."""

    def __init__(self, lhs: Any, connector: str, rhs: Any):
        self.lhs = lhs if isinstance(lhs, Expression) else Value(lhs)
        self.connector = connector
        self.rhs = rhs if isinstance(rhs, Expression) else Value(rhs)

    def resolve_expression(
        self,
        query=None,
        allow_joins: bool = True,
        reuse=None,
        summarize: bool = False,
        placeholder: str = "?",
        **kwargs,
    ) -> Tuple[str, List[Any]]:
        lhs_sql, lhs_params = self.lhs.resolve_expression(
            query=query,
            allow_joins=allow_joins,
            reuse=reuse,
            summarize=summarize,
            placeholder=placeholder,
            **kwargs,
        )
        rhs_sql, rhs_params = self.rhs.resolve_expression(
            query=query,
            allow_joins=allow_joins,
            reuse=reuse,
            summarize=summarize,
            placeholder=placeholder,
            **kwargs,
        )
        
        # Enclose in parentheses for safety if needed
        sql = f"({lhs_sql} {self.connector} {rhs_sql})"
        params = lhs_params + rhs_params
        return sql, params

    def __repr__(self):
        return f"({self.lhs!r} {self.connector} {self.rhs!r})"


class UnaryExpression(Expression):
    """Applies a unary operator (- or +) to an expression."""

    def __init__(self, op: str, expression: Any):
        self.op = op
        self.expression = expression if isinstance(expression, Expression) else Value(expression)

    def resolve_expression(
        self,
        query=None,
        allow_joins: bool = True,
        reuse=None,
        summarize: bool = False,
        placeholder: str = "?",
        **kwargs,
    ) -> Tuple[str, List[Any]]:
        sql, params = self.expression.resolve_expression(
            query=query,
            allow_joins=allow_joins,
            reuse=reuse,
            summarize=summarize,
            placeholder=placeholder,
            **kwargs,
        )
        return f"{self.op}({sql})", params

    def __repr__(self):
        return f"{self.op}({self.expression!r})"
