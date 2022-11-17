from BooleanModel import Boolean_Model

model = Boolean_Model("./sample/*")

# Para que una expresion sea correcta hay que escribir todos los tokens separados por espacio (por ahora)
# Los tokens son : AND, OR, not,(,) y los terminos
# Ej: not not (study AND not viscous )
print(model.proces_query("not ( study AND not ( viscous ) )"))









