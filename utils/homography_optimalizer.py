import numpy as np
import torch.optim

from scipy.optimize import minimize

def reprojection_error(h_chain, points1, points2):
    points1_h = np.hstack([points1, np.ones((points1.shape[0], 1))])  # Shape: (n, 3)
    points1_proj_h = np.einsum('nij,nj->ni', h_chain, points1_h)  # Shape: (n, 3)

    # cartesian
    points1_proj = points1_proj_h[:, :2] / points1_proj_h[:, 2, None]  # Shape: (n, 2)

    # reprojection errors
    errors = np.linalg.norm(points1_proj - points2, axis=1)  # Shape: (n,)
    return errors


def objective(h_flat, correspondences_flat):
    num_homographies = len(h_flat) // 9
    h_matrices = h_flat.reshape((num_homographies, 3, 3))  # Shape: (num_homographies, 3, 3)

    # correspondences
    points1 = np.array([c[0] for c in correspondences_flat])  # Shape: (n, 2)
    points2 = np.array([c[1] for c in correspondences_flat])  # Shape: (n, 2)
    homography1_indices = np.array([c[2] for c in correspondences_flat])  # Shape: (n,)
    homography2_indices = np.array([c[3] for c in correspondences_flat])  # Shape: (n,)

    # homographies
    h1 = h_matrices[homography1_indices]  # Shape: (n, 3, 3)
    h2 = h_matrices[homography2_indices]  # Shape: (n, 3, 3)

    # chained homographies
    h1_inv = np.linalg.inv(h1)  # Shape: (n, 3, 3)
    h_chain = np.einsum('nij,njk->nik', h2, h1_inv)  # Chain h2 and h1_inv, shape: (n, 3, 3)

    # reprojection errors
    errors = reprojection_error(h_chain, points1, points2)  # Shape: (n,)
    total_error = np.sum(errors)

    return total_error


def homography_constraints(h_flat):
    h_matrices = [h_flat[i:i + 9].reshape(3, 3) for i in range(0, len(h_flat), 9)]
    return [np.linalg.det(H) for H in h_matrices]  # One constraint per homography


def extract_initial_affine_from_homography(H):
    # extract translation
    tx, ty = H[0, 2], H[1, 2]

    # extract rotation and scaling
    A = H[:2, :2]
    singular_values = np.linalg.svd(A, compute_uv=False)
    s = np.mean(singular_values)
    R = A / s
    theta = np.arctan2(R[1, 0], R[0, 0])

    return np.array([s, theta, tx, ty])


def objective_affine(affines_flat, correspondences_flat):
    # reshape the flattened parameters into a list of [s, theta, tx, ty]
    num_affines = len(affines_flat) // 4
    affines = affines_flat.reshape((num_affines, 4))

    # extract s R t
    s = affines[:, 0]
    theta = affines[:, 1]
    tx = affines[:, 2]
    ty = affines[:, 3]

    # compute R stack
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrices = np.stack([
        np.stack([cos_theta, -sin_theta], axis=-1),
        np.stack([sin_theta, cos_theta], axis=-1)
    ], axis=-2)  # Shape: (num_affines, 2, 2)

    # compute affine matrices
    affine_matrices = s[:, None, None] * rotation_matrices  # Shape: (num_affines, 2, 2)

    # stack translations
    translations = np.stack([tx, ty], axis=-1)  # Shape: (num_affines, 2)

    # extract correspondences
    p = np.array([c[0] for c in correspondences_flat])  # Shape: (num_correspondences, 2)
    q = np.array([c[1] for c in correspondences_flat])  # Shape: (num_correspondences, 2)
    aff_idx1 = np.array([c[2] for c in correspondences_flat])  # Shape: (num_correspondences,)
    aff_idx2 = np.array([c[3] for c in correspondences_flat])  # Shape: (num_correspondences,)

    A1 = affine_matrices[aff_idx1]  # Shape: (num_correspondences, 2, 2)
    T1 = translations[aff_idx1]  # Shape: (num_correspondences, 2)
    A2 = affine_matrices[aff_idx2]  # Shape: (num_correspondences, 2, 2)
    T2 = translations[aff_idx2]  # Shape: (num_correspondences, 2)

    # scaling and rotation
    A1_inv = np.linalg.inv(A1)  # Shape: (num_correspondences, 2, 2)

    # chain transformations
    A_chain = np.einsum('nij,njk->nik', A2, A1_inv)  # p.dot(A2, inv(A1))
    T_chain = T2 - np.einsum('nij,nj->ni', A_chain, T1)  # translation

    # apply chained transformation to p
    p_transformed = np.einsum('nij,nj->ni', A_chain, p) + T_chain  # Shape: (num_correspondences, 2)

    # compute reprojection errors
    errors = np.linalg.norm(p_transformed - q, axis=1)  # Shape: (num_correspondences,)
    total_error = np.sum(errors)

    return total_error


class HomographyOptimizer:
    def __init__(self, max_matches=200, optimization_model='homography', silent=False):
        self.max_matches = max_matches
        self.silent = silent
        self.optimization_model = optimization_model

    def optimize(self, pairs, homographies, correspondences):
        if self.optimization_model == 'affine':
            return self.optimize_affine(pairs, homographies, correspondences)
        elif self.optimization_model == 'homography':
            return self.optimize_homography(pairs, homographies, correspondences)
        else:
            raise ValueError('Invalid optimization model')

    def optimize_affine(self, pairs, homographies, correspondences):
        if not self.silent:
            print('Optimizing affine...')
        # extract affine transformations
        affines = [extract_initial_affine_from_homography(H) for H in homographies]
        affines_flat = np.array(affines).flatten()

        # flatten correspondences
        correspondences_flat = []

        # Iterate over corresponding_points to extract and flatten data
        for correspondence in correspondences:
            pair = correspondence['pair']
            points = correspondence['points']

            # Extract homography indices
            affine1 = pair[0]
            affine2 = pair[1]

            # Pair up corresponding points from the two arrays
            points1 = points[0]
            points2 = points[1]

            for p, q in zip(points1, points2):
                # Append the flattened correspondence data
                correspondences_flat.append((p, q, affine1, affine2))

        # subsample correspondences
        # print('Number of correspondences:', len(correspondences_flat))
        subsample_factor = (int)(max(1, len(correspondences_flat) / (self.max_matches * len(homographies))))
        correspondences_flat = correspondences_flat[::subsample_factor]

        # print size of correspondences
        # print('Number of correspondences:', len(correspondences_flat))
        if not self.silent:
            print('Initial error:', objective_affine(affines_flat, correspondences_flat))


        result = minimize(
            objective_affine,
            affines_flat,
            args=(correspondences_flat,),
            # method='trust-constr',
            method='BFGS',
            options={'disp': not self.silent}
        )
        # result = minimize(objective, h_flat, args=(correspondences_flat,))
        if not self.silent:
            print('Final error:', objective_affine(result.x, correspondences_flat))

        optimized_affines = result.x.reshape((len(affines), 4))
        affine_matrices = []

        for s, theta, tx, ty in optimized_affines:
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            # Construct the 3x3 affine matrix
            affine_matrix = np.array([
                [s * cos_theta, -s * sin_theta, tx],
                [s * sin_theta, s * cos_theta, ty],
                [0, 0, 1]
            ])
            affine_matrices.append(affine_matrix)

        # normalize the affine matrices
        A_fixed = affine_matrices[pairs[0][0]]
        A_fixed_inv = np.linalg.inv(A_fixed)
        affine_matrices_normalized = [A @ A_fixed_inv for A in affine_matrices]

        # Return the normalized affine matrices
        return affine_matrices_normalized

    def own_LBFGS(self, x, correspondeces):
        x = torch.tensor(x, requires_grad=True, device="cuda")
        optimizer = torch.optim.LBFGS([x], lr=1.0, max_iter=100)

        def closure():
            optimizer.zero_grad()
            loss = objective(x, correspondeces)  # Pass extra arguments
            loss.backward()
            return loss

        for i in range(10):
            optimizer.step(closure)

        return x
    def optimize_homography(self, pairs, homographies, correspondences):
        if not self.silent:
            print('Optimizing homography...')
        # flatten homographies
        h_flat = np.array([h.flatten() for h in homographies]).flatten()

        # flatten correspondences
        correspondences_flat = []
        # for homography_index, (points1, points2) in enumerate(correspondences):
        #    for p, q in zip(points1, points2):
        #        correspondences_flat.append((p, q, homography1, homography2))

        # Iterate over corresponding_points to extract and flatten data
        for correspondence in correspondences:
            pair = correspondence['pair']
            points = correspondence['points']

            # Extract homography indices
            homography1 = pair[0]
            homography2 = pair[1]

            # Pair up corresponding points from the two arrays
            points1 = points[0]
            points2 = points[1]

            for p, q in zip(points1, points2):
                # Append the flattened correspondence data
                correspondences_flat.append((p, q, homography1, homography2))

        # subsample correspondences
        # print('Number of correspondences:', len(correspondences_flat))
        subsample_factor = (int)(max(1, len(correspondences_flat) / (self.max_matches * len(homographies))))
        correspondences_flat = correspondences_flat[::subsample_factor]

        # print size of correspondences
        # print('Number of correspondences:', len(correspondences_flat))
        if not self.silent:
            print('Initial error:', objective(h_flat, correspondences_flat))

        constraints = [
            {'type': 'ineq', 'fun': lambda h_flat, idx=i: np.linalg.det(h_flat[idx * 9:(idx + 1) * 9].reshape(3, 3))}
            for i in range(len(homographies))]
        # todo: find solver that supports constraints

        #result = self.own_LBFGS(h_flat, correspondences_flat)
        # result = minimize(
        #     objective,
        #     h_flat,
        #     args=(correspondences_flat,),
        #     # method='trust-constr',
        #     method='BFGS',
        #     options={'disp': not self.silent}
        # )
        result = minimize(objective, h_flat, args=(correspondences_flat,))
        if not self.silent:
            print('Final error:', objective(result.x, correspondences_flat))

        optimized_h = [result.x[i:i + 9].reshape(3, 3) for i in range(0, len(result.x), 9)]

        # normalize the H matrices
        H_fixed = optimized_h[pairs[0][0]]
        H_fixed_inv = np.linalg.inv(H_fixed)
        h_matrices_normalized = [H @ H_fixed_inv for H in optimized_h]

        return h_matrices_normalized