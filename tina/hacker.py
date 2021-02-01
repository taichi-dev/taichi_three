import taichi as ti


hasattr(ti, '_tinahacked') or setattr(ti, '_tinahacked', 1) or setattr(ti,
        'static', lambda x, *xs: [x] + list(xs) if xs else x) or setattr(
        ti.Matrix, 'element_wise_writeback_binary', (lambda f: lambda x, y, z:
        (y.__name__ != 'assign' or not setattr(y, '__name__', '_assign'))
        and f(x, y, z))(ti.Matrix.element_wise_writeback_binary)) or setattr(
        ti.Matrix, 'is_global', (lambda f: lambda x: len(x) and f(x))(
        ti.Matrix.is_global)) or setattr(ti.TaichiOperations, '__pos__',
        lambda x: x) or setattr(ti, 'pi', __import__('math').pi) or setattr(ti,
        'tau', __import__('math').tau) or setattr(ti, 'materialize_callback',
        (lambda f: lambda x: [(x() if ti.get_runtime().materialized else f(x)),
        x][1])(ti.materialize_callback)) or setattr(ti, 'expr_init', (lambda f:
        lambda x: x if isinstance(x, dict) else f(x))(ti.expr_init)) or setattr(
        ti, 'expr_init_func', (lambda f: lambda x: x if isinstance(x, dict)
        else f(x))(ti.expr_init_func)) or print('[Tina] Taichi properties hacked')


@eval('lambda x: x()')
def _():
    class GUI(ti.GUI):
        def __init__(self, name='Tina', res=512, **kwargs):
            if isinstance(res, ti.Matrix):
                res = res.entries
            if isinstance(res, list):
                res = tuple(res)
            super().__init__(name=name, res=res, **kwargs)
            self._post_show_cbs = []

        def post_show(self, cb):
            self._post_show_cbs.append(cb)
            return cb

        def rects(self, topleft, bottomright, radius=1, color=0xffffff):
            import numpy as np
            topright = np.stack([topleft[:, 0], bottomright[:, 1]], axis=1)
            bottomleft = np.stack([bottomright[:, 0], topleft[:, 1]], axis=1)
            self.lines(topleft, topright, radius, color)
            self.lines(topright, bottomright, radius, color)
            self.lines(bottomright, bottomleft, radius, color)
            self.lines(bottomleft, topleft, radius, color)

        def show(self, *args, **kwargs):
            super().show(*args, **kwargs)
            for cb in self._post_show_cbs:
                cb(self)

    ti.GUI = GUI


@eval('lambda x: x()')
def _():
    if hasattr(ti, 'smart'):
        return

    ti.smart = lambda x: x

    import copy, ast
    from taichi.lang.transformer import ASTTransformerBase, ASTTransformerPreprocess

    old_get_decorator = ASTTransformerBase.get_decorator

    @staticmethod
    def get_decorator(node):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and isinstance(
                    node.func.value, ast.Name) and node.func.value.id == 'ti'
                and node.func.attr in ['smart']):
            return old_get_decorator(node)
        return node.func.attr

    ASTTransformerBase.get_decorator = get_decorator

    old_visit_struct_for = ASTTransformerPreprocess.visit_struct_for

    def visit_struct_for(self, node, is_grouped):
        if not is_grouped:
            decorator = self.get_decorator(node.iter)
            if decorator == 'smart':  # so smart!
                self.current_control_scope().append('smart')
                self.generic_visit(node, ['body'])
                t = self.parse_stmt('if 1: pass; del a')
                t.body[0] = node
                target = copy.deepcopy(node.target)
                target.ctx = ast.Del()
                if isinstance(target, ast.Tuple):
                    for tar in target.elts:
                        tar.ctx = ast.Del()
                t.body[-1].targets = [target]
                return t

        return old_visit_struct_for(self, node, is_grouped)

    ASTTransformerPreprocess.visit_struct_for = visit_struct_for


__all__ = []
