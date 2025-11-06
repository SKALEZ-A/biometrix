import { Router } from 'express';
import { TransactionController } from '../controllers/transaction.controller';
import { authMiddleware } from '../middleware/auth.middleware';
import { validationMiddleware } from '../middleware/validation.middleware';

const router = Router();
const transactionController = new TransactionController();

router.post(
  '/transactions',
  authMiddleware,
  transactionController.createTransaction.bind(transactionController)
);

router.get(
  '/transactions/:id',
  authMiddleware,
  transactionController.getTransaction.bind(transactionController)
);

router.get(
  '/transactions',
  authMiddleware,
  transactionController.getTransactions.bind(transactionController)
);

router.post(
  '/transactions/:id/verify',
  authMiddleware,
  transactionController.verifyTransaction.bind(transactionController)
);

router.post(
  '/transactions/:id/flag',
  authMiddleware,
  transactionController.flagTransaction.bind(transactionController)
);

router.get(
  '/transactions/user/:userId',
  authMiddleware,
  transactionController.getUserTransactions.bind(transactionController)
);

router.get(
  '/transactions/risk/analysis',
  authMiddleware,
  transactionController.getRiskAnalysis.bind(transactionController)
);

export default router;
