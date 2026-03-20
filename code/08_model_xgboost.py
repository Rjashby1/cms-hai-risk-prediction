def run_xgboost_cv(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    results = []

    pos = np.sum(y == 1)
    neg = np.sum(y == 0)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n[FOLD {fold}]")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=neg / pos,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=-1
        )

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        # Probabilities
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Threshold tuning
        threshold = find_best_threshold(y_test, y_prob)
        y_pred = (y_prob >= threshold).astype(int)

        # Metrics
        roc = roc_auc_score(y_test, y_prob)

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)

        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        mcc = matthews_corrcoef(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print(f"ROC-AUC: {roc:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")
        print(f"F1 (macro): {f1_macro:.4f}")
        print(f"F1 (weighted): {f1_weighted:.4f}")
        print(f"MCC: {mcc:.4f}")
        print(f"Threshold: {threshold:.4f}")
        print("Confusion Matrix:\n", cm)

        results.append({
            "roc_auc": roc,
            "pr_auc": pr_auc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "mcc": mcc,
            "threshold": threshold
        })

    results_df = pd.DataFrame(results)

    print("\n=== CV SUMMARY ===")
    print(results_df.mean())
    print("\nSTD:")
    print(results_df.std())

    return results_df
