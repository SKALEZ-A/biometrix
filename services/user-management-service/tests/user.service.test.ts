import { UserService } from '../src/services/user.service';
import { User } from '../src/models/user.model';

describe('UserService', () => {
  let userService: UserService;

  beforeEach(() => {
    userService = new UserService();
  });

  describe('createUser', () => {
    it('should create a new user with valid data', async () => {
      const userData = {
        email: 'test@example.com',
        passwordHash: 'password123',
        firstName: 'John',
        lastName: 'Doe',
      };

      const user = await userService.createUser(userData);
      
      expect(user).toBeDefined();
      expect(user.email).toBe(userData.email);
      expect(user.firstName).toBe(userData.firstName);
      expect(user.isVerified).toBe(false);
    });

    it('should throw error for duplicate email', async () => {
      const userData = {
        email: 'duplicate@example.com',
        passwordHash: 'password123',
        firstName: 'John',
        lastName: 'Doe',
      };

      await userService.createUser(userData);
      
      await expect(userService.createUser(userData)).rejects.toThrow('Email already exists');
    });

    it('should hash password during creation', async () => {
      const userData = {
        email: 'hash@example.com',
        passwordHash: 'plainPassword',
        firstName: 'John',
        lastName: 'Doe',
      };

      const user = await userService.createUser(userData);
      
      expect(user.passwordHash).not.toBe('plainPassword');
      expect(user.passwordHash.length).toBeGreaterThan(20);
    });
  });

  describe('getUserById', () => {
    it('should return user when found', async () => {
      const userData = {
        email: 'find@example.com',
        passwordHash: 'password123',
        firstName: 'Jane',
        lastName: 'Smith',
      };

      const createdUser = await userService.createUser(userData);
      const foundUser = await userService.getUserById(createdUser.userId);
      
      expect(foundUser).toBeDefined();
      expect(foundUser?.userId).toBe(createdUser.userId);
    });

    it('should return null when user not found', async () => {
      const user = await userService.getUserById('non-existent-id');
      expect(user).toBeNull();
    });
  });

  describe('updateUser', () => {
    it('should update user information', async () => {
      const userData = {
        email: 'update@example.com',
        passwordHash: 'password123',
        firstName: 'Original',
        lastName: 'Name',
      };

      const user = await userService.createUser(userData);
      const updated = await userService.updateUser(user.userId, {
        firstName: 'Updated',
      });
      
      expect(updated?.firstName).toBe('Updated');
      expect(updated?.lastName).toBe('Name');
    });

    it('should not allow password update through updateUser', async () => {
      const userData = {
        email: 'password@example.com',
        passwordHash: 'password123',
        firstName: 'Test',
        lastName: 'User',
      };

      const user = await userService.createUser(userData);
      const originalHash = user.passwordHash;
      
      await userService.updateUser(user.userId, {
        passwordHash: 'newPassword',
      } as any);
      
      const updated = await userService.getUserById(user.userId);
      expect(updated?.passwordHash).toBe(originalHash);
    });
  });

  describe('changePassword', () => {
    it('should change password with valid current password', async () => {
      const userData = {
        email: 'changepass@example.com',
        passwordHash: 'oldPassword123',
        firstName: 'Test',
        lastName: 'User',
      };

      const user = await userService.createUser(userData);
      const result = await userService.changePassword(
        user.userId,
        'oldPassword123',
        'newPassword456'
      );
      
      expect(result).toBe(true);
    });

    it('should fail with invalid current password', async () => {
      const userData = {
        email: 'wrongpass@example.com',
        passwordHash: 'correctPassword',
        firstName: 'Test',
        lastName: 'User',
      };

      const user = await userService.createUser(userData);
      
      await expect(
        userService.changePassword(user.userId, 'wrongPassword', 'newPassword')
      ).rejects.toThrow('Invalid current password');
    });
  });

  describe('verifyEmail', () => {
    it('should verify email with valid token', async () => {
      const userData = {
        email: 'verify@example.com',
        passwordHash: 'password123',
        firstName: 'Test',
        lastName: 'User',
      };

      const user = await userService.createUser(userData);
      const result = await userService.verifyEmail(user.emailVerificationToken!);
      
      expect(result).toBe(true);
      
      const verified = await userService.getUserById(user.userId);
      expect(verified?.isVerified).toBe(true);
    });

    it('should fail with invalid token', async () => {
      const result = await userService.verifyEmail('invalid-token');
      expect(result).toBe(false);
    });
  });
});
