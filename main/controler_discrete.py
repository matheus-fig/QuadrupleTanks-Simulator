class ControladorPIDiscretoEuler:
    def __init__(self, Kp, Ti, Ts):
        """
        Inicializa o controlador PI discreto usando Euler para frente.

        Args:
            Kp: Ganho proporcional.
            Ti: Tempo integral.
            Ts: Período de amostragem.
        """
        self.Kp = Kp
        self.Ti = Ti
        self.Ts = Ts
        self.Ki = Kp / Ti
        self.integral = 0  # Inicializa a integral do erro

    def calcular_saida(self, erro):
        """
        Calcula a saída do controlador PI discreto.

        Args:
            erro: Erro atual.

        Returns:
            Saída do controlador.
        """
        # Ação proporcional
        proporcional = self.Kp * erro

        # Ação integral (Euler para frente)
        self.integral += erro * self.Ts
        integral = self.Ki * self.integral

        # Saída do controlador
        saida = proporcional + integral
        return saida

# Exemplo de uso
Kp = 1.0  # Ganho proporcional
Ti = 0.1  # Tempo integral
Ts = 0.01 # Período de amostragem

controlador = ControladorPIDiscretoEuler(Kp, Ti, Ts)

# # Simulação (exemplo com erro constante)
# erro = 1.0
# for i in range(100):
#     saida = controlador.calcular_saida(erro)
#     print(f"Tempo: {i*Ts:.2f}, Saída: {saida:.2f}")