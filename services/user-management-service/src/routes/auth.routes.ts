import { Router } from 'express';
import { AuthController } from '../controllers/auth.controller';

const router = Router();
const authController = new AuthController();

router.post('/register', authController.register.bind(authController));
router.post('/login', authController.login.bind(authController));
router.post('/logout', authController.logout.bind(authController));
router.post('/refresh-token', authController.refreshToken.bind(authController));
router.post('/verify-email', authController.verifyEmail.bind(authController));
router.post('/forgot-password', authController.forgotPassword.bind(authController));
router.post('/reset-password', authController.resetPassword.bind(authController));
router.post('/verify-2fa', authController.verify2FA.bind(authController));
router.post('/enable-2fa', authController.enable2FA.bind(authController));
router.post('/disable-2fa', authController.disable2FA.bind(authController));

export { router as authRoutes };
